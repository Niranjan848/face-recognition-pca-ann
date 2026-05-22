"""
Face Recognition System: PCA + ANN (Backpropagation)
=====================================================
Implements the Turk & Pentland (1991) Eigenfaces approach combined
with a custom Artificial Neural Network for classification.

Dataset: ORL/AT&T Face Database
  https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip

Usage:
  python face_recognition_pca_ann.py --dataset ./dataset
  python face_recognition_pca_ann.py --dataset ./dataset --k_values 5 10 20 50 100
"""

import os
import sys
import argparse
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Load and Generate Face Database
# ============================================================

def load_face_database(dataset_path, image_size=(92, 112)):
    """
    Load all face images from the dataset directory structure.
    Supports ORL/AT&T format: dataset/<subject>/<image>.pgm

    Returns
    -------
    face_db   : ndarray (mn x p) — each column is a flattened face image
    labels    : ndarray (p,)     — integer class label per image
    label_names: list            — folder names (subject IDs)
    """
    face_db = []
    labels  = []
    label_names = []

    subject_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not subject_dirs:
        raise ValueError(
            f"No subdirectories found in '{dataset_path}'. "
            "Each subject should have its own folder."
        )

    for label_idx, subject in enumerate(subject_dirs):
        subject_path = os.path.join(dataset_path, subject)
        label_names.append(subject)

        image_files = sorted([
            f for f in os.listdir(subject_path)
            if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png', '.bmp'))
        ])

        for img_file in image_files:
            img = cv2.imread(
                os.path.join(subject_path, img_file),
                cv2.IMREAD_GRAYSCALE
            )
            if img is None:
                continue
            img = cv2.resize(img, image_size)          # (rows, cols)
            face_db.append(img.flatten().astype(np.float64))
            labels.append(label_idx)

    if not face_db:
        raise ValueError("No images could be loaded from the dataset path.")

    face_db = np.column_stack(face_db)   # (mn x p)
    labels  = np.array(labels)

    return face_db, labels, label_names


# ============================================================
# STEP 2 & 3: Mean Calculation and Mean Zero
# ============================================================

def compute_mean_and_zero(face_db):
    """
    Compute the mean face M and subtract it from every image column.

    M_i = (1/p) * sum_j  FaceDb(i, j)      (mn x 1)
    A(i)_mn×p = FaceDb(i)_mn×p − M_mn×1    (broadcast)

    Returns
    -------
    A : ndarray (mn x p) — mean-zero face matrix
    M : ndarray (mn x 1) — mean face vector
    """
    M = face_db.mean(axis=1, keepdims=True)   # (mn x 1)
    A = face_db - M                            # (mn x p)
    return A, M


# ============================================================
# STEP 4: Surrogate Covariance  (Turk & Pentland, 1991)
# ============================================================

def compute_surrogate_covariance(A):
    """
    Standard covariance C = A·Aᵀ has shape (mn × mn) — infeasible for images.
    Turk & Pentland's surrogate: C_surr = Aᵀ·A  has shape (p × p).

    The non-zero eigenvalues and the valid directions are identical;
    only the zero-eigenvalue directions differ.

    Returns
    -------
    C : ndarray (p x p) — surrogate covariance matrix
    """
    return A.T @ A     # (p x p)


# ============================================================
# STEP 5: Eigenvalue / Eigenvector Decomposition
# ============================================================

def eigen_decomposition(C):
    """
    Compute eigenvalues and eigenvectors of the surrogate covariance.
    Results are sorted in descending eigenvalue order.

    Returns
    -------
    eigenvalues  : ndarray (p,)
    eigenvectors : ndarray (p x p)  — columns are eigenvectors
    """
    eigenvalues, eigenvectors = eigh(C)          # eigh is stable for symmetric matrices
    idx = np.argsort(eigenvalues)[::-1]          # descending order
    return eigenvalues[idx], eigenvectors[:, idx]


# ============================================================
# STEP 6 & 7: Feature Vectors and Eigenfaces
# ============================================================

def generate_eigenfaces(A, eigenvectors, k):
    """
    Select k best eigenvectors and project mean-zero faces to get eigenfaces.

    Φ_(k×mn) = Ψᵀ_(k×p)  ·  Aᵀ_(p×mn)

    Parameters
    ----------
    A           : (mn x p) mean-zero face matrix
    eigenvectors: (p  x p) eigenvectors of surrogate covariance
    k           : number of principal components to keep

    Returns
    -------
    Phi : ndarray (k x mn) — eigenfaces (each row is one eigenface)
    """
    Psi_k = eigenvectors[:, :k]        # (p x k) — top-k eigenvectors
    Phi   = Psi_k.T @ A.T             # (k x mn)
    return Phi


# ============================================================
# STEP 8: Generate Signatures
# ============================================================

def generate_signatures(Phi, A):
    """
    Project each mean-zero training face onto the eigenface space
    to obtain its compact signature (feature vector).

    ω_(k×p) = Φ_(k×mn) · A_(mn×p)

    Returns
    -------
    omega : ndarray (k x p) — signature matrix; column i is signature of face i
    """
    return Phi @ A    # (k x p)


# ============================================================
# STEP 9: Backpropagation ANN
# ============================================================

class ANN:
    """
    Fully-connected feedforward neural network with backpropagation.

    Architecture: input → [hidden layers] → softmax output
    Hidden layers use sigmoid activation.
    Loss: cross-entropy.
    Optimizer: mini-batch SGD with Xavier initialization.
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 learning_rate=0.01):
        self.lr     = learning_rate
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

        # Xavier weight initialization
        self.W = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            * np.sqrt(2.0 / layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.b = [np.zeros((1, s)) for s in layer_sizes[1:]]

    # ----------------------------------------------------------
    # Activations
    # ----------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_grad(z):
        s = ANN._sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def _softmax(z):
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def _forward(self, X):
        self._activations = [X]
        self._pre_acts    = []

        current = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = current @ W + b
            self._pre_acts.append(z)
            if i < len(self.W) - 1:
                current = self._sigmoid(z)       # hidden layers
            else:
                current = self._softmax(z)       # output layer
            self._activations.append(current)

        return current   # predicted probabilities

    # ----------------------------------------------------------
    # Backward pass (backpropagation)
    # ----------------------------------------------------------
    def _backward(self, y_onehot):
        m   = y_onehot.shape[0]
        dW  = [None] * len(self.W)
        db  = [None] * len(self.b)

        # Output layer gradient (cross-entropy + softmax combined)
        delta = self._activations[-1] - y_onehot          # (m x out)

        for i in reversed(range(len(self.W))):
            dW[i] = self._activations[i].T @ delta / m
            db[i] = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._sigmoid_grad(self._pre_acts[i - 1])

        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def fit(self, X, y, epochs=500, batch_size=32, verbose=False):
        """
        Train the network.

        Parameters
        ----------
        X       : (n x k) input signatures
        y       : (n,)    integer class labels
        epochs  : number of training epochs
        batch_size : mini-batch size
        verbose : print loss every 100 epochs
        """
        n_classes  = len(np.unique(y))
        y_onehot   = np.eye(n_classes)[y]
        history    = []

        for epoch in range(1, epochs + 1):
            # Shuffle
            perm = np.random.permutation(X.shape[0])
            X_s, y_s = X[perm], y_onehot[perm]

            # Mini-batch SGD
            for start in range(0, X.shape[0], batch_size):
                Xb = X_s[start: start + batch_size]
                yb = y_s[start: start + batch_size]
                self._forward(Xb)
                self._backward(yb)

            # Log loss
            out  = self._forward(X)
            loss = -np.mean(np.sum(y_onehot * np.log(out + 1e-12), axis=1))
            history.append(loss)

            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(out, axis=1) == y)
                print(f"    Epoch {epoch:4d} | loss={loss:.4f} | train_acc={acc:.3f}")

        return history

    def predict(self, X):
        """Return predicted class indices."""
        probs = self._forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Return predicted probabilities."""
        return self._forward(X)


# ============================================================
# PCA Face Recognition System  (wrapper)
# ============================================================

class PCAFaceRecognition:
    """
    Full PCA + ANN face recognition pipeline.

    Training
    --------
    1. Build face database matrix  (mn × p)
    2. Compute mean face M
    3. Mean-zero: A = FaceDb − M
    4. Surrogate covariance  C = Aᵀ·A
    5. Eigen decomposition of C
    6. Select k best eigenvectors → Feature matrix Ψ
    7. Eigenfaces Φ = Ψᵀ · Aᵀ
    8. Signatures ω = Φ · A
    9. Train ANN on (ωᵀ, labels)

    Testing
    -------
    1. Read test image I, flatten to column vector
    2. Mean-zero: I₂ = I − M
    3. Project: Ω = Φ · I₂  (k × 1)
    4. ANN.predict(Ωᵀ) → predicted label
    """

    def __init__(self, k=50, image_size=(92, 112),
                 hidden_sizes=None, ann_lr=0.01, ann_epochs=500,
                 ann_batch_size=32):
        self.k            = k
        self.image_size   = image_size
        self.hidden_sizes = hidden_sizes   # None = auto
        self.ann_lr       = ann_lr
        self.ann_epochs   = ann_epochs
        self.ann_batch_size = ann_batch_size

        # Artifacts set during fit()
        self.M           = None
        self.Phi         = None
        self.ann         = None
        self.label_names = None
        self.n_classes   = None

    # ----------------------------------------------------------
    def fit(self, face_db, labels, label_names, verbose=False):
        """
        Train the full PCA + ANN pipeline.

        Parameters
        ----------
        face_db     : (mn x p) training face matrix
        labels      : (p,) integer class labels
        label_names : list of subject names
        """
        p = face_db.shape[1]
        self.label_names = label_names
        self.n_classes   = len(np.unique(labels))
        k_eff            = min(self.k, p - 1)   # k cannot exceed p-1

        if k_eff != self.k:
            print(f"    [WARNING] k reduced from {self.k} to {k_eff} (only {p} training images)")
            self.k = k_eff

        # Steps 2–3: mean and mean-zero
        A, self.M = compute_mean_and_zero(face_db)

        # Step 4: surrogate covariance
        C = compute_surrogate_covariance(A)

        # Step 5: eigendecomposition
        eigenvalues, eigenvectors = eigen_decomposition(C)

        # Steps 6–7: eigenfaces
        self.Phi = generate_eigenfaces(A, eigenvectors, self.k)

        # Step 8: signatures
        omega = generate_signatures(self.Phi, A)   # (k x p)

        # Step 9: train ANN
        X_ann = omega.T   # (p x k) — each row = one signature

        h = self.hidden_sizes
        if h is None:
            h = [max(64, self.n_classes * 4), max(32, self.n_classes * 2)]

        self.ann = ANN(
            input_size=self.k,
            hidden_sizes=h,
            output_size=self.n_classes,
            learning_rate=self.ann_lr
        )
        self.ann.fit(X_ann, labels,
                     epochs=self.ann_epochs,
                     batch_size=self.ann_batch_size,
                     verbose=verbose)
        return self

    # ----------------------------------------------------------
    def _project(self, face_col_vector):
        """
        Project a single face (mn x 1) to eigenspace.
        Returns omega (k,).
        """
        I2    = face_col_vector.reshape(-1, 1) - self.M   # mean-zero
        omega = self.Phi @ I2                              # (k x 1)
        return omega.flatten()

    def _get_signatures(self, face_db):
        """Get signatures matrix (n_test x k) for a face_db (mn x n_test)."""
        n = face_db.shape[1]
        sigs = np.zeros((n, self.k))
        for i in range(n):
            sigs[i] = self._project(face_db[:, i])
        return sigs

    # ----------------------------------------------------------
    def predict(self, face_db_test):
        """
        Predict labels for test faces.

        Parameters
        ----------
        face_db_test : (mn x q) test face matrix

        Returns
        -------
        predictions : (q,) integer predicted labels
        """
        sigs = self._get_signatures(face_db_test)
        return self.ann.predict(sigs)

    def predict_proba(self, face_db_test):
        sigs = self._get_signatures(face_db_test)
        return self.ann.predict_proba(sigs)

    def evaluate(self, face_db_test, labels_test):
        """
        Returns accuracy (0–1) and predicted labels.
        """
        preds = self.predict(face_db_test)
        acc   = np.mean(preds == labels_test)
        return acc, preds

    # ----------------------------------------------------------
    def reconstruction_error(self, face_db):
        """
        Compute per-face reconstruction error in eigenspace.
        High error → face is far from the training distribution → likely imposter.

        Returns
        -------
        errors : (q,) reconstruction error for each face
        """
        q      = face_db.shape[1]
        errors = np.zeros(q)

        for i in range(q):
            face_vec = face_db[:, i].reshape(-1, 1)
            I2       = face_vec - self.M                   # mean-zero
            omega    = self.Phi @ I2                       # (k x 1)  — project
            I2_hat   = self.Phi.T @ omega                  # (mn x 1) — reconstruct
            errors[i] = np.linalg.norm(I2 - I2_hat)

        return errors


# ============================================================
# Train / Test Split  (per-subject, stratified)
# ============================================================

def stratified_split(face_db, labels, train_ratio=0.6, seed=42):
    """
    Split face_db into train/test sets keeping ~train_ratio images
    per subject in the training set.
    """
    rng       = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1

    train_idx, test_idx = [], []

    for cls in range(n_classes):
        idx   = np.where(labels == cls)[0]
        idx   = rng.permutation(idx)
        n_tr  = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:n_tr])
        test_idx.extend(idx[n_tr:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    return (face_db[:, train_idx], labels[train_idx],
            face_db[:, test_idx],  labels[test_idx])


# ============================================================
# Experiment (a): Accuracy vs k
# ============================================================

def experiment_accuracy_vs_k(face_db_train, labels_train,
                              face_db_test, labels_test,
                              label_names, k_values,
                              ann_epochs=500, ann_lr=0.01,
                              output_dir="."):
    """
    Train a separate model for each k in k_values and evaluate accuracy.
    Generates and saves a plot.

    Returns
    -------
    results : dict  {k: accuracy_percent}
    best_k  : int
    """
    print("\n" + "=" * 55)
    print("Experiment (a): Classification Accuracy vs k")
    print("=" * 55)

    n_subjects = len(label_names)
    max_k      = face_db_train.shape[1] - 1
    valid_ks   = [k for k in k_values if k <= max_k]

    if not valid_ks:
        raise ValueError(
            f"All k values exceed maximum allowed ({max_k}). "
            f"Use smaller k values."
        )

    skipped = set(k_values) - set(valid_ks)
    if skipped:
        print(f"  [INFO] Skipping k={skipped} (exceed training set size {max_k})")

    results = {}

    for k in valid_ks:
        print(f"\n  k = {k:4d}", end=" | ")
        model = PCAFaceRecognition(
            k=k, ann_lr=ann_lr, ann_epochs=ann_epochs
        )
        model.fit(face_db_train, labels_train, label_names, verbose=False)
        acc, _ = model.evaluate(face_db_test, labels_test)
        results[k] = acc * 100
        print(f"Accuracy = {acc*100:.2f}%")

    # ----- Plot -----
    ks   = list(results.keys())
    accs = list(results.values())
    best_k = ks[int(np.argmax(accs))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, accs, 'o-', color='royalblue', linewidth=2.5,
            markersize=9, markerfacecolor='white', markeredgewidth=2)
    ax.axvline(best_k, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Best k = {best_k}  ({max(accs):.1f}%)')

    for k, a in zip(ks, accs):
        ax.annotate(f'{a:.1f}%', (k, a),
                    textcoords='offset points', xytext=(0, 11),
                    ha='center', fontsize=8.5, color='#333333')

    ax.set_xlabel('Number of Eigenfaces  (k)', fontsize=13)
    ax.set_ylabel('Recognition Accuracy  (%)', fontsize=13)
    ax.set_title('PCA + ANN Face Recognition:  Accuracy vs k', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, 'accuracy_vs_k.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [Saved] {plot_path}")

    return results, best_k


# ============================================================
# Experiment (b): Imposter Detection
# ============================================================

def experiment_imposter_detection(face_db, labels, label_names,
                                  n_imposter_subjects=8,
                                  k=None, ann_epochs=500,
                                  ann_lr=0.01, train_ratio=0.6,
                                  output_dir="."):
    """
    Designate a subset of subjects as 'imposters' (not enrolled).
    Train on enrolled subjects only, then test recognition + imposter rejection.

    Imposter detection strategy: reconstruction error threshold.
    Faces with high reconstruction error are flagged as imposters.
    """
    print("\n" + "=" * 55)
    print("Experiment (b): Imposter Detection")
    print("=" * 55)

    n_subjects = len(label_names)
    n_enrolled = n_subjects - n_imposter_subjects

    if n_enrolled < 2:
        raise ValueError("Too many imposter subjects; too few enrolled subjects remain.")

    enrolled_ids = list(range(n_enrolled))
    imposter_ids = list(range(n_enrolled, n_subjects))

    print(f"  Enrolled subjects : {n_enrolled}  (labels 0–{n_enrolled-1})")
    print(f"  Imposter subjects : {n_imposter_subjects}  (labels {n_enrolled}–{n_subjects-1})")

    # Split enrolled-only data for training
    enrolled_mask = np.isin(labels, enrolled_ids)
    face_db_enr   = face_db[:, enrolled_mask]
    labels_enr    = labels[enrolled_mask]
    enr_names     = [label_names[i] for i in enrolled_ids]

    (face_db_tr, lbl_tr,
     face_db_val, lbl_val) = stratified_split(face_db_enr, labels_enr,
                                               train_ratio=train_ratio)

    # If k not given, use a reasonable default
    max_k = face_db_tr.shape[1] - 1
    k_use = min(k if k else 30, max_k)
    print(f"  Using k = {k_use}")

    # Train on enrolled subjects
    model = PCAFaceRecognition(k=k_use, ann_lr=ann_lr, ann_epochs=ann_epochs)
    model.fit(face_db_tr, lbl_tr, enr_names, verbose=False)

    # Validation: enrolled test faces
    acc_enrolled, preds_val = model.evaluate(face_db_val, lbl_val)
    print(f"\n  Enrolled test accuracy: {acc_enrolled*100:.2f}%")

    # Imposter test faces (all images of imposter subjects)
    imp_mask      = np.isin(labels, imposter_ids)
    face_db_imp   = face_db[:, imp_mask]
    n_imp_images  = face_db_imp.shape[1]
    print(f"  Imposter test images: {n_imp_images}")

    # Reconstruction errors for both groups
    err_genuine  = model.reconstruction_error(face_db_val)    # enrolled
    err_imposter = model.reconstruction_error(face_db_imp)    # imposters

    # Find optimal threshold (maximise balanced accuracy)
    all_errs = np.concatenate([err_genuine, err_imposter])
    all_true = np.concatenate([
        np.zeros(len(err_genuine)),   # 0 = genuine
        np.ones(len(err_imposter))    # 1 = imposter
    ])

    thresholds = np.linspace(all_errs.min(), all_errs.max(), 200)
    best_thr, best_bacc = thresholds[0], 0.0

    for thr in thresholds:
        pred_imp = (all_errs > thr).astype(int)
        tp  = np.sum((pred_imp == 1) & (all_true == 1))
        tn  = np.sum((pred_imp == 0) & (all_true == 0))
        fp  = np.sum((pred_imp == 1) & (all_true == 0))
        fn  = np.sum((pred_imp == 0) & (all_true == 1))
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        bacc = (tpr + tnr) / 2
        if bacc > best_bacc:
            best_bacc  = bacc
            best_thr   = thr

    # Final metrics at best threshold
    genuine_rejected  = np.sum(err_genuine  > best_thr)   # FAR (False Accept Rate inverted)
    imposter_rejected = np.sum(err_imposter > best_thr)   # TAR on imposters

    far = genuine_rejected  / len(err_genuine)             # False Acceptance Rate
    frr = (len(err_imposter) - imposter_rejected) / len(err_imposter)  # False Rejection Rate

    print(f"\n  Optimal threshold  : {best_thr:.2f}")
    print(f"  Balanced Accuracy  : {best_bacc*100:.2f}%")
    print(f"  Genuine accepted   : {len(err_genuine)-genuine_rejected}/{len(err_genuine)}"
          f"  (FAR = {far*100:.1f}%)")
    print(f"  Imposters rejected : {imposter_rejected}/{len(err_imposter)}"
          f"  (FRR = {frr*100:.1f}%)")

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of reconstruction errors
    ax = axes[0]
    bins = np.linspace(all_errs.min(), all_errs.max(), 30)
    ax.hist(err_genuine,  bins=bins, alpha=0.7, color='steelblue',
            label='Genuine (enrolled)')
    ax.hist(err_imposter, bins=bins, alpha=0.7, color='tomato',
            label='Imposter (not enrolled)')
    ax.axvline(best_thr, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {best_thr:.0f}')
    ax.set_xlabel('Reconstruction Error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Reconstruction Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: ROC-style accuracy vs threshold
    ax2 = axes[1]
    tar_list, far_list = [], []
    for thr in thresholds:
        rej_imp = np.sum(err_imposter > thr) / len(err_imposter)
        rej_gen = np.sum(err_genuine  > thr) / len(err_genuine)
        tar_list.append(rej_imp)
        far_list.append(rej_gen)

    ax2.plot(far_list, tar_list, color='purple', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    op_far = np.sum(err_genuine  > best_thr) / len(err_genuine)
    op_tar = np.sum(err_imposter > best_thr) / len(err_imposter)
    ax2.scatter([op_far], [op_tar], color='red', zorder=5, s=100,
                label=f'Operating point\n(FAR={op_far:.2f}, TAR={op_tar:.2f})')
    ax2.set_xlabel('False Acceptance Rate  (FAR)', fontsize=12)
    ax2.set_ylabel('Imposter Rejection Rate (TAR)', fontsize=12)
    ax2.set_title('ROC Curve — Imposter Detection', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'imposter_detection.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [Saved] {plot_path}")

    return {
        'threshold'        : best_thr,
        'balanced_accuracy': best_bacc,
        'far'              : far,
        'frr'              : frr,
    }


# ============================================================
# Visualise Eigenfaces
# ============================================================

def visualise_eigenfaces(face_db_train, labels_train, label_names,
                          k=20, image_size=(92, 112), output_dir="."):
    """Save a grid of the first k eigenfaces as a PNG."""
    A, M = compute_mean_and_zero(face_db_train)
    C    = compute_surrogate_covariance(A)
    _, V = eigen_decomposition(C)
    Phi  = generate_eigenfaces(A, V, k)   # (k x mn)

    rows = int(np.ceil(k / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(12, rows * 2.5))
    axes = axes.flatten()

    for i in range(k):
        ef = Phi[i].reshape(image_size[::-1])   # (rows, cols)
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        axes[i].imshow(ef, cmap='gray')
        axes[i].set_title(f'EF {i+1}', fontsize=8)
        axes[i].axis('off')

    for i in range(k, len(axes)):
        axes[i].axis('off')

    # Also show mean face
    mean_img = M.reshape(image_size[::-1])
    mean_img = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
    axes[0].imshow(mean_img, cmap='gray')
    axes[0].set_title('Mean Face', fontsize=8)

    fig.suptitle(f'Eigenfaces (top {k} components)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'eigenfaces.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='PCA + ANN Face Recognition (Turk & Pentland 1991)'
    )
    p.add_argument('--dataset', required=True,
                   help='Path to the extracted dataset folder')
    p.add_argument('--image_size', nargs=2, type=int, default=[92, 112],
                   metavar=('W', 'H'),
                   help='Resize images to W×H (default: 92 112)')
    p.add_argument('--k_values', nargs='+', type=int,
                   default=[5, 10, 20, 30, 50, 75, 100],
                   help='k values to evaluate in experiment (a)')
    p.add_argument('--ann_epochs', type=int, default=500,
                   help='ANN training epochs (default: 500)')
    p.add_argument('--ann_lr', type=float, default=0.01,
                   help='ANN learning rate (default: 0.01)')
    p.add_argument('--train_ratio', type=float, default=0.6,
                   help='Fraction of data used for training (default: 0.6)')
    p.add_argument('--n_imposters', type=int, default=8,
                   help='Number of subjects held out as imposters (default: 8)')
    p.add_argument('--output_dir', default='./output',
                   help='Directory to save plots and results')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    image_size = tuple(args.image_size)   # (W, H)

    print("\n" + "=" * 55)
    print("  PCA + ANN Face Recognition System")
    print("  Turk & Pentland (1991) Eigenfaces")
    print("=" * 55)

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    print(f"\n[1] Loading dataset from: {args.dataset}")
    face_db, labels, label_names = load_face_database(args.dataset, image_size)
    p = face_db.shape[1]
    mn = face_db.shape[0]
    n_subjects = len(label_names)

    print(f"    Images loaded   : {p}")
    print(f"    Subjects        : {n_subjects}")
    print(f"    Image dims (WxH): {image_size}  →  flattened = {mn}")

    # ----------------------------------------------------------
    # Train / Test split
    # ----------------------------------------------------------
    print(f"\n[2] Splitting (train={args.train_ratio:.0%}, test={1-args.train_ratio:.0%})")
    face_db_tr, lbl_tr, face_db_te, lbl_te = stratified_split(
        face_db, labels, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"    Train : {face_db_tr.shape[1]} images")
    print(f"    Test  : {face_db_te.shape[1]} images")

    # ----------------------------------------------------------
    # Visualise eigenfaces
    # ----------------------------------------------------------
    print("\n[3] Visualising eigenfaces ...")
    k_vis = min(20, face_db_tr.shape[1] - 1)
    visualise_eigenfaces(face_db_tr, lbl_tr, label_names,
                          k=k_vis, image_size=image_size,
                          output_dir=args.output_dir)

    # ----------------------------------------------------------
    # Experiment (a): accuracy vs k
    # ----------------------------------------------------------
    results_a, best_k = experiment_accuracy_vs_k(
        face_db_tr, lbl_tr, face_db_te, lbl_te,
        label_names,
        k_values=args.k_values,
        ann_epochs=args.ann_epochs,
        ann_lr=args.ann_lr,
        output_dir=args.output_dir
    )

    # ----------------------------------------------------------
    # Experiment (b): imposter detection
    # ----------------------------------------------------------
    n_imp = min(args.n_imposters, n_subjects - 2)
    results_b = experiment_imposter_detection(
        face_db, labels, label_names,
        n_imposter_subjects=n_imp,
        k=best_k,
        ann_epochs=args.ann_epochs,
        ann_lr=args.ann_lr,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Best k                : {best_k}")
    print(f"  Best accuracy         : {max(results_a.values()):.2f}%")
    print(f"  Imposter threshold    : {results_b['threshold']:.2f}")
    print(f"  Imposter detection BA : {results_b['balanced_accuracy']*100:.2f}%")
    print(f"  FAR                   : {results_b['far']*100:.2f}%")
    print(f"  FRR                   : {results_b['frr']*100:.2f}%")
    print(f"\n  Output plots saved to : {args.output_dir}/")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()
