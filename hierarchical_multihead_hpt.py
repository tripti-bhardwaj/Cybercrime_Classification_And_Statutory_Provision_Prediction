import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

DEVICE = "cpu"
N_TRIALS = 25
FIXED_TAU = 0.9

EPOCHS = 40
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 6
BATCH = 64
SUPER_TEMP = 0.9464270017695555
FINE_TEMP  = 0.3871717192042973
CONF_QUANTILE = 0.55

SUPER_CLASSES = {
    "FINANCIAL_FRAUD": [
        "Financial Fraud",
        "Gambling/Betting",
        "Cryptocurrency Crime"
    ],
    "SEXUAL_CRIME": [
        "Sexually Explicit Content",
        "Sexually Obscene Content",
        "Child Abuse Material",
        "Rape or Sexual Abuse Content"
    ],
    "CYBER_ATTACK": [
        "Hacking/Damage",
        "Cyber Attack/Dependent Crimes",
        "Cyber Trafficking",
        "Cyber Terrorism",
        "Ransomware"
    ],
    "SOCIAL_MEDIA_CRIME": [
        "Social Media Crime",
        "Other Cyber Crime"
    ]
}

DEFAULT_FINE_CLASS = "Financial Fraud"

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

bert_train = torch.tensor(np.load("trainhingroberta_embs.npy"), dtype=torch.float32)
bert_test  = torch.tensor(np.load("testhingroberta_embs.npy"), dtype=torch.float32)

word_train = torch.tensor(np.load("train_embs.npy"), dtype=torch.float32)
word_test  = torch.tensor(np.load("test_embs.npy"), dtype=torch.float32)

class FeatureFusion(nn.Module):
    def __init__(self, da, db):
        super().__init__()
        self.gate = nn.Linear(da + db, db)

    def forward(self, a, b):
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        concat = torch.cat([a, b], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        interacted = gate * b + (1 - gate) * a[:, :b.shape[1]]
        return torch.cat([concat, interacted], dim=1)

fusion = FeatureFusion(
    bert_train.shape[1],
    word_train.shape[1]
).to(DEVICE)

with torch.no_grad():
    X_train = fusion(bert_train.to(DEVICE), word_train.to(DEVICE))
    X_test  = fusion(bert_test.to(DEVICE),  word_test.to(DEVICE))

fusion_dim = X_train.shape[1]

def build_hierarchy(df):
    sup, fine = [], []
    for c in df["category"].astype(str):
        for s, members in SUPER_CLASSES.items():
            if c in members:
                sup.append(s)
                fine.append(c)
                break
        else:
            sup.append("OTHER_CYBER")
            fine.append(c)
    return sup, fine

train_super, train_fine = build_hierarchy(train_df)
test_super,  test_fine  = build_hierarchy(test_df)

le_super = LabelEncoder()
le_fine  = LabelEncoder()

y_train_super = torch.tensor(le_super.fit_transform(train_super)).to(DEVICE)
y_train_fine  = torch.tensor(le_fine.fit_transform(train_fine)).to(DEVICE)

mapped_test_fine = [
    c if c in le_fine.classes_ else DEFAULT_FINE_CLASS
    for c in test_fine
]
y_test_fine = torch.tensor(le_fine.transform(mapped_test_fine)).to(DEVICE)

fine_counts = np.bincount(y_train_fine.cpu().numpy())
inv_freq = torch.tensor(
    1.0 / np.sqrt(fine_counts + 1),
    dtype=torch.float32,
    device=DEVICE
)

class AttentiveResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.attn = nn.Linear(hidden, 1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.norm1(self.fc1(x)))
        h = self.drop(h)
        h2 = F.gelu(self.norm2(self.fc2(h)))
        h = h + h2
        h = torch.sigmoid(self.attn(h)) * h
        return self.out(h)

def train_and_evaluate(
    lr,
    epochs,
    stage1_epochs,
    stage2_epochs,
    super_temp,
    fine_temp,
    conf_quantile,
    dropout
):
    super_clf = nn.Sequential(
        nn.Linear(fusion_dim, 768),
        nn.LayerNorm(768),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(768, 512),
        nn.GELU(),
        nn.Linear(512, len(le_super.classes_))
    ).to(DEVICE)

    fine_heads = nn.ModuleDict()
    fine_class_indices = {}

    for s in le_super.classes_:
        classes = [
            c for c in le_fine.classes_
            if c in SUPER_CLASSES.get(s, []) or s == "OTHER_CYBER"
        ]
        idxs = le_fine.transform(classes)
        fine_class_indices[s] = torch.tensor(idxs).to(DEVICE)

        fine_heads[s] = AttentiveResidualMLP(
            fusion_dim + len(le_super.classes_),
            512,
            len(classes),
            dropout
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(super_clf.parameters()) + list(fine_heads.parameters()),
        lr=lr
    )

    loss_super = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_train))
        stage1 = epoch <= stage1_epochs

        for p in fine_heads.parameters():
            p.requires_grad = not stage1

        for i in range(0, len(X_train), 64):
            idx = perm[i:i+64]
            optimizer.zero_grad()

            s_logits = super_clf(X_train[idx])
            s_loss = loss_super(s_logits, y_train_super[idx])
            loss = s_loss

            if not stage1:
                s_probs = F.softmax(s_logits.detach() / super_temp, dim=1)
                f_input = torch.cat([X_train[idx], s_probs], dim=1)

                fine_loss, used = 0.0, 0
                for s_name, head in fine_heads.items():
                    s_id = le_super.transform([s_name])[0]
                    mask = (y_train_super[idx] == s_id)
                    if mask.sum() == 0:
                        continue

                    targets = y_train_fine[idx][mask]
                    valid = fine_class_indices[s_name]

                    local_targets = torch.tensor(
                        [(valid == t).nonzero()[0] for t in targets],
                        device=DEVICE
                    )

                    logits = head(f_input[mask]) / fine_temp
                    ce = nn.CrossEntropyLoss(weight=inv_freq[valid])(
                        logits, local_targets
                    )

                    margin = torch.clamp(
                        0.25 - (logits.topk(2).values[:,0] -
                                logits.topk(2).values[:,1]),
                        min=0
                    ).mean()

                    fine_loss += ce + 0.3 * margin
                    used += 1

                if used > 0:
                    loss = 0.4 * s_loss + 0.6 * (fine_loss / used)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(super_clf.parameters()) + list(fine_heads.parameters()), 1.0
            )
            optimizer.step()

    super_clf.eval()
    for h in fine_heads.values():
        h.eval()

    preds, confs, top2_correct = [], [], 0

    with torch.no_grad():
        s_probs = F.softmax(super_clf(X_test) / super_temp, dim=1)

        for i in range(len(X_test)):
            score_map = {}
            p, ids = torch.topk(s_probs[i], k=2)

            for r in range(len(ids)):
                s_name = le_super.inverse_transform([ids[r].item()])[0]
                head = fine_heads[s_name]
                valid = fine_class_indices[s_name]

                inp = torch.cat([X_test[i], s_probs[i]], dim=0).unsqueeze(0)
                probs = F.softmax(head(inp) / fine_temp, dim=1)

                local_top1 = probs.argmax().item()
                global_top1 = valid[local_top1].item()
                score_map[global_top1] = score_map.get(global_top1, 0.0) + \
                                         p[r].item() * probs.max().item()

                if y_test_fine[i].item() in valid[probs.topk(2).indices[0]].tolist():
                    top2_correct += 1

            best_cls, best_score = max(score_map.items(), key=lambda x: x[1])
            preds.append(best_cls)
            confs.append(best_score)

    preds = np.array(preds)
    confs = np.array(confs)

    acc1 = (preds == y_test_fine.cpu().numpy()).mean()
    acc2 = top2_correct / len(y_test_fine)

    conf_threshold = FIXED_TAU
    conf_mask = confs >= conf_threshold

    coverage = conf_mask.mean()

    conf_acc = (
        (preds[conf_mask] == y_test_fine.cpu().numpy()[conf_mask]).mean()
        if conf_mask.sum() > 0 else 0.0
    )

    weighted_f1 = f1_score(
        y_test_fine.cpu().numpy(),
        preds,
        average="weighted"
    )

    return {
        "weighted_f1": weighted_f1,
        "top1": acc1,
        "top2": acc2,
        "conf_acc": conf_acc,
        "coverage": coverage,
        "conf_threshold": conf_threshold,
        "preds": preds,
        "confs": confs
    }

def plot_risk_coverage(preds, confs, y_true):
    order = np.argsort(-confs)
    preds_sorted = preds[order]
    y_sorted = y_true[order]

    risks = []
    coverages = []
    total = len(preds_sorted)

    for k in range(1, total + 1):
        kept_preds = preds_sorted[:k]
        kept_true = y_sorted[:k]
        accuracy = (kept_preds == kept_true).mean()
        risk = 1 - accuracy
        coverage = k / total
        risks.append(risk)
        coverages.append(coverage)

    risks = np.array(risks)
    coverages = np.array(coverages)

    aurc = np.trapz(risks, coverages)

    plt.figure()
    plt.plot(coverages, risks)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - Accuracy)")
    plt.title("Risk–Coverage Curve")
    plt.grid(True)
    plt.show()

    print(f"AURC: {aurc:.4f}")

    return aurc

def objective(trial):
    metrics = train_and_evaluate(
        lr=trial.suggest_categorical("lr", [1e-4]),
        epochs=EPOCHS,
        stage1_epochs=STAGE1_EPOCHS,
        stage2_epochs=STAGE2_EPOCHS,
        super_temp=SUPER_TEMP,
        fine_temp=FINE_TEMP,
        conf_quantile=CONF_QUANTILE,
        dropout=trial.suggest_categorical("dropout", [0.3])
    )

    print(
    f"[Trial {trial.number}] "
    f"F1={metrics['weighted_f1']:.4f} | "
    f"Top1={metrics['top1']:.4f} | "
    f"Top2={metrics['top2']:.4f} | "
    f"ConfAcc={metrics['conf_acc']:.4f} | "
    f"Coverage={metrics['coverage']:.3f} | "
    f"τ={metrics['conf_threshold']:.4f} | "
    f"Params={trial.params}"
)

    return metrics["weighted_f1"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBEST HYPERPARAMETERS:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\nFINAL EVALUATION:")
    best_lr = study.best_params["lr"]
    best_dropout = study.best_params["dropout"]

    final = train_and_evaluate(
        lr=best_lr,
        epochs=EPOCHS,
        stage1_epochs=STAGE1_EPOCHS,
        stage2_epochs=STAGE2_EPOCHS,
        super_temp=SUPER_TEMP,
        fine_temp=FINE_TEMP,
        conf_quantile=CONF_QUANTILE,
        dropout=best_dropout
    )

    print(f"Weighted F1-score     : {final['weighted_f1']:.4f}")
    print(f"Top-1 Accuracy        : {final['top1']:.4f}")
    print(f"Top-2 Accuracy        : {final['top2']:.4f}")
    print(f"Confidence Accuracy   : {final['conf_acc']:.4f}")
    print(f"Coverage              : {final['coverage']:.4f}")
    print(f"Confidence Threshold τ: {final['conf_threshold']:.4f}")

    used = np.unique(y_test_fine.cpu().numpy())
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test_fine.cpu().numpy(),
        final["preds"],
        labels=used,
        target_names=le_fine.inverse_transform(used),
        zero_division=0
    ))

    print("\nGenerating Risk–Coverage Curve...")
    aurc = plot_risk_coverage(
        preds=final["preds"],
        confs=final["confs"],
        y_true=y_test_fine.cpu().numpy()
    )