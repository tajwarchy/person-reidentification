# src/side_by_side_export.py
import os
import pickle
import yaml
import numpy as np
import cv2
from PIL import Image


# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        import yaml
        return yaml.safe_load(f)


# ── Draw one side-by-side panel ────────────────────────────
def make_panel(crop_bgr, gallery_path, gid, pid, score,
               panel_w=320, panel_h=256):
    """
    Left  : detection crop from video
    Right : top gallery match
    Returns a BGR numpy panel of size (panel_h, panel_w*2+divider)
    """
    divider = 4
    canvas  = np.zeros((panel_h, panel_w * 2 + divider, 3), dtype=np.uint8)
    canvas[:, panel_w:panel_w + divider] = (80, 80, 80)  # grey divider

    def paste(img_bgr, col_offset):
        resized = cv2.resize(img_bgr, (panel_w, panel_h))
        canvas[:, col_offset:col_offset + panel_w] = resized

    # Left: video crop
    paste(crop_bgr, 0)

    # Right: gallery image
    g_img = cv2.imread(gallery_path)
    if g_img is not None:
        paste(g_img, panel_w + divider)

    # Labels
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2

    # Left label
    cv2.rectangle(canvas, (0, 0), (panel_w, 24), (30, 30, 30), -1)
    cv2.putText(canvas, f"Video  Global ID:{gid}",
                (4, 17), font, font_scale, (255, 255, 255), thickness)

    # Right label
    rx = panel_w + divider
    cv2.rectangle(canvas, (rx, 0), (rx + panel_w, 24), (30, 30, 30), -1)
    cv2.putText(canvas, f"Gallery PID:{pid}  sim:{score:.3f}",
                (rx + 4, 17), font, font_scale, (100, 255, 100), thickness)

    return canvas


# ── Build video ────────────────────────────────────────────
def build_sidebyside_video(tracking_state_path, output_path,
                           n_persons, fps, panel_w=320, panel_h=256):
    with open(tracking_state_path, "rb") as f:
        state = pickle.load(f)

    counts = state["global_id_counts"]
    crops  = state["global_id_crops"]

    # Top N most frequently re-identified persons
    top_gids = sorted(counts.keys(), key=lambda g: -counts[g])[:n_persons]
    # Filter to only those with crop data
    top_gids = [g for g in top_gids if g in crops]
    print(f"Top {len(top_gids)} re-identified persons:")
    for g in top_gids:
        print(f"  Global ID {g}: {counts[g]} frames | "
              f"Gallery PID {crops[g]['pid']} | "
              f"sim {crops[g]['score']:.3f}")

    if not top_gids:
        print("❌ No crop data found — re-run video_demo.py first")
        return

    # Stack panels vertically into one frame
    panels = []
    for g in top_gids:
        info  = crops[g]
        panel = make_panel(
            info["crop"],
            info["gallery"],
            gid=g,
            pid=info["pid"],
            score=info["score"],
            panel_w=panel_w,
            panel_h=panel_h,
        )
        panels.append(panel)

    # Pad to n_persons panels if fewer found
    while len(panels) < n_persons:
        panels.append(np.zeros_like(panels[0]))

    frame  = np.vstack(panels)
    fh, fw = frame.shape[:2]
    print(f"Output frame size: {fw}x{fh}")

    # Write as 5-second video (static grid held for duration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc  = cv2.VideoWriter_fourcc(*"avc1")
    writer  = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    n_frames = int(fps * 5)   # 5 seconds
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()

    # Also save as PNG for README
    png_path = output_path.replace(".mp4", ".png")
    cv2.imwrite(png_path, frame)
    print(f"✅ Side-by-side video saved : {output_path}")
    print(f"✅ Side-by-side PNG saved   : {png_path}")


# ── Main ───────────────────────────────────────────────────
def main(args):
    cfg = load_config(args.config)

    tracking_state_path = f"{cfg['paths']['output_videos']}/tracking_state.pkl"
    output_path         = cfg["video"]["output_sidebyside"]
    n_persons           = cfg["video"]["top_reidentified_persons"]
    fps                 = cfg["video"]["fps_output"]

    build_sidebyside_video(
        tracking_state_path,
        output_path,
        n_persons,
        fps,
        panel_w=320,
        panel_h=256,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid_config.yaml")
    args   = parser.parse_args()
    main(args)