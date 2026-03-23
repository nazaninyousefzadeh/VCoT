import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/SalChartQA/unified_approved.csv"
df = pd.read_csv(CSV_PATH)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# 1. Distribution of chart types (image_type)
print("=== Chart Type Distribution ===")
print(df["image_type"].value_counts())
fig, ax = plt.subplots(figsize=(8, 4))
df["image_type"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Chart Type Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_chart_type.png", dpi=150)
plt.close()

# 2. Simple vs Not Simple
print("\n=== Simple vs Not Simple ===")
print(df["is_chart_simple"].value_counts())
fig, ax = plt.subplots(figsize=(5, 4))
df["is_chart_simple"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Simple vs Not Simple Charts")
ax.set_ylabel("Count")
ax.set_xticklabels(["Not Simple", "Simple"], rotation=0)
plt.tight_layout()
plt.savefig("data/eda_simple.png", dpi=150)
plt.close()

# 3. Question type distribution
print("\n=== Question Type Distribution ===")
print(df["question_type"].value_counts())
fig, ax = plt.subplots(figsize=(8, 4))
df["question_type"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Question Type Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_question_type.png", dpi=150)
plt.close()

# 4. Question type per chart type
print("\n=== Question Type per Chart Type ===")
ct = pd.crosstab(df["image_type"], df["question_type"])
print(ct)
fig, ax = plt.subplots(figsize=(10, 5))
ct.plot(kind="bar", ax=ax)
ax.set_title("Question Type per Chart Type")
ax.set_ylabel("Count")
plt.legend(title="Question Type", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("data/eda_qtype_per_chart.png", dpi=150)
plt.close()

# 5. Numerical vs Non-numerical answers
print("\n=== Numerical vs Non-numerical Answers ===")
print(df["is_answer_numerical"].value_counts())
fig, ax = plt.subplots(figsize=(5, 4))
df["is_answer_numerical"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Numerical vs Non-numerical Answers")
ax.set_ylabel("Count")
ax.set_xticklabels(["Numerical", "Non-numerical"], rotation=0)
plt.tight_layout()
plt.savefig("data/eda_numerical.png", dpi=150)
plt.close()

# 6. Correct vs Incorrect
print("\n=== Correct vs Incorrect ===")
print(df["is_correct"].value_counts())
fig, ax = plt.subplots(figsize=(5, 4))
df["is_correct"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Correct vs Incorrect")
ax.set_ylabel("Count")
ax.set_xticklabels(["Correct", "Incorrect"], rotation=0)
plt.tight_layout()
plt.savefig("data/eda_correct.png", dpi=150)
plt.close()

# 7. Average number of clicks per chart type and per question type
print("\n=== Avg Clicks per Chart Type ===")
avg_chart = df.groupby("image_type")["number_of_clicks"].mean().sort_values(ascending=False)
print(avg_chart.round(2))
fig, ax = plt.subplots(figsize=(8, 4))
avg_chart.plot(kind="bar", ax=ax)
ax.set_title("Avg Clicks per Chart Type")
ax.set_ylabel("Avg Clicks")
plt.tight_layout()
plt.savefig("data/eda_clicks_chart.png", dpi=150)
plt.close()

print("\n=== Avg Clicks per Question Type ===")
avg_q = df.groupby("question_type")["number_of_clicks"].mean().sort_values(ascending=False)
print(avg_q.round(2))
fig, ax = plt.subplots(figsize=(8, 4))
avg_q.plot(kind="bar", ax=ax)
ax.set_title("Avg Clicks per Question Type")
ax.set_ylabel("Avg Clicks")
plt.tight_layout()
plt.savefig("data/eda_clicks_qtype.png", dpi=150)
plt.close()

# 8. Avg clicks for simple vs non-simple
print("\n=== Avg Clicks: Simple vs Not Simple ===")
avg_simple = df.groupby("is_chart_simple")["number_of_clicks"].mean()
print(avg_simple.round(2))
fig, ax = plt.subplots(figsize=(5, 4))
avg_simple.plot(kind="bar", ax=ax)
ax.set_title("Avg Clicks: Simple vs Not Simple")
ax.set_ylabel("Avg Clicks")
ax.set_xticklabels(["Not Simple", "Simple"], rotation=0)
plt.tight_layout()
plt.savefig("data/eda_clicks_simple.png", dpi=150)
plt.close()

# 9. Simple vs Not Simple per chart type
print("\n=== Simple vs Not Simple per Chart Type ===")
ct_simple_chart = pd.crosstab(df["image_type"], df["is_chart_simple"])
ct_simple_chart.columns = ["Not Simple", "Simple"]
print(ct_simple_chart)
fig, ax = plt.subplots(figsize=(8, 4))
ct_simple_chart.plot(kind="bar", ax=ax)
ax.set_title("Simple vs Not Simple per Chart Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_simple_per_chart.png", dpi=150)
plt.close()

# 10. Simple vs Not Simple per question type
print("\n=== Simple vs Not Simple per Question Type ===")
ct_simple_q = pd.crosstab(df["question_type"], df["is_chart_simple"])
ct_simple_q.columns = ["Not Simple", "Simple"]
print(ct_simple_q)
fig, ax = plt.subplots(figsize=(8, 4))
ct_simple_q.plot(kind="bar", ax=ax)
ax.set_title("Simple vs Not Simple per Question Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_simple_per_qtype.png", dpi=150)
plt.close()

# 11. Faceted heatmap: error rate by chart type x question type, split by simple/not simple
import numpy as np

print("\n=== Error Rate: Chart Type x Question Type x Simple ===")
for label, is_simple in [("Simple", True), ("Not Simple", False)]:
    subset = df[df["is_chart_simple"] == is_simple]
    total = subset.groupby(["image_type", "question_type"]).size().unstack(fill_value=0)
    wrong = subset[subset["is_correct"] == False].groupby(["image_type", "question_type"]).size().unstack(fill_value=0)
    rate = (wrong / total * 100).fillna(0).round(1)
    print(f"\n{label} — Error Rate (%):")
    print(rate)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, (label, is_simple) in zip(axes, [("Simple", True), ("Not Simple", False)]):
    subset = df[df["is_chart_simple"] == is_simple]
    total = subset.groupby(["image_type", "question_type"]).size().unstack(fill_value=0)
    wrong = subset[subset["is_correct"] == False].groupby(["image_type", "question_type"]).size().unstack(fill_value=0)
    wrong = wrong.reindex(index=total.index, columns=total.columns, fill_value=0)
    rate = (wrong / total * 100).fillna(0).round(1)

    im = ax.imshow(rate.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(len(rate.columns)))
    ax.set_xticklabels(rate.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rate.index)))
    ax.set_yticklabels(rate.index)
    ax.set_title(f"{label} Charts")

    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            val = rate.values[i, j]
            count = int(wrong.values[i, j]) if not np.isnan(wrong.values[i, j]) else 0
            if total.values[i, j] > 0:
                ax.text(j, i, f"{val:.0f}%\n({count})", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=axes, label="Error Rate (%)", shrink=0.8)
fig.suptitle("Error Rate by Chart Type × Question Type", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("data/eda_error_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# 12. Samples without answer
missing = df["answer"].isna().sum()
empty = (df["answer"].astype(str).str.strip() == "").sum()
print(f"\n=== Samples Without Answer ===")
print(f"NaN answers: {missing}")
print(f"Empty string answers: {empty}")
print(f"Total missing: {missing + empty}")

print("\nDone! Plots saved to data/ folder.")
