"""
smoke_test.py — Verify the PCA + ANN pipeline with a tiny synthetic dataset.
Run this before pointing the main script at the real face dataset.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import tempfile, shutil

# ── Build a tiny fake dataset ──────────────────────────────────────────────
def make_fake_dataset(n_subjects=10, images_per_subject=10,
                      img_w=46, img_h=56, seed=0):
    rng  = np.random.default_rng(seed)
    root = tempfile.mkdtemp(prefix='face_test_')

    for s in range(n_subjects):
        sdir = os.path.join(root, f's{s+1}')
        os.makedirs(sdir)
        # Each subject: a random base image + small noise per shot
        base = rng.integers(30, 220, (img_h, img_w), dtype=np.uint8)
        for i in range(images_per_subject):
            noise = rng.integers(-15, 15, base.shape, dtype=np.int16)
            img   = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sdir, f'{i+1}.pgm'), img)

    return root, (img_w, img_h)


# ── Run test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building synthetic dataset …")
    dataset_path, image_size = make_fake_dataset()

    try:
        from face_recognition_pca_ann import (
            load_face_database, stratified_split,
            experiment_accuracy_vs_k, experiment_imposter_detection,
            visualise_eigenfaces,
        )

        out_dir = tempfile.mkdtemp(prefix='face_out_')
        face_db, labels, label_names = load_face_database(dataset_path, image_size)

        print(f"Loaded: {face_db.shape[1]} images | {len(label_names)} subjects "
              f"| flattened size = {face_db.shape[0]}")

        face_db_tr, lbl_tr, face_db_te, lbl_te = stratified_split(
            face_db, labels, train_ratio=0.6
        )
        print(f"Train: {face_db_tr.shape[1]}  Test: {face_db_te.shape[1]}")

        visualise_eigenfaces(face_db_tr, lbl_tr, label_names,
                              k=min(10, face_db_tr.shape[1]-1),
                              image_size=image_size, output_dir=out_dir)

        results_a, best_k = experiment_accuracy_vs_k(
            face_db_tr, lbl_tr, face_db_te, lbl_te,
            label_names, k_values=[2, 4, 6, 8],
            ann_epochs=200, ann_lr=0.01, output_dir=out_dir
        )

        results_b = experiment_imposter_detection(
            face_db, labels, label_names,
            n_imposter_subjects=2, k=best_k,
            ann_epochs=200, ann_lr=0.01,
            train_ratio=0.6, output_dir=out_dir
        )

        print(f"\n✓ Smoke test passed!  Best k={best_k}, "
              f"best acc={max(results_a.values()):.1f}%")
        print(f"  Output plots: {out_dir}/")

    finally:
        shutil.rmtree(dataset_path, ignore_errors=True)
