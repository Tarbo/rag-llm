"""
SKU Availability Forecasting: PR Curve with OTIF Score Analysis

This script demonstrates how to find the optimal confidence threshold by analyzing:
- Precision-Recall curve at different thresholds
- OTIF (On-Time In-Full) scores at each threshold
- Trade-offs between model performance and business metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("="*70)
print("SKU AVAILABILITY FORECASTING: PR CURVE WITH OTIF ANALYSIS")
print("="*70)

# Define confidence thresholds from 50% to 100% in steps of 5%
thresholds = np.arange(50, 105, 5)

# Generate realistic Precision values (increases with threshold)
# At lower thresholds, more FPs → lower precision
precision = np.array([
    0.65, 0.68, 0.72, 0.76, 0.80, 0.84, 0.87, 0.90, 0.93, 0.95, 0.97
])

# Generate realistic Recall values (decreases with threshold)
# At higher thresholds, more FNs → lower recall
recall = np.array([
    0.95, 0.92, 0.89, 0.85, 0.81, 0.76, 0.70, 0.63, 0.55, 0.45, 0.35
])

# Generate F1 scores
f1_score = 2 * (precision * recall) / (precision + recall)

# Generate OTIF scores
# OTIF is hurt by False Positives (predicting available when not)
# Lower thresholds → More FPs → Lower OTIF
# Higher thresholds → Fewer FPs → Higher OTIF
otif_score = np.array([
    0.72, 0.75, 0.79, 0.83, 0.87, 0.90, 0.92, 0.93, 0.94, 0.93, 0.92
])

# Simulate number of False Positives and False Negatives
total_samples = 1000
true_positives = (recall * precision * total_samples * 0.5).astype(int)
false_positives = ((1 - precision) * total_samples * 0.5 / precision).astype(int)
false_negatives = ((1 - recall) * total_samples * 0.5 / recall).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precision,
    'Recall': recall,
    'F1_Score': f1_score,
    'OTIF': otif_score,
    'False_Positives': false_positives,
    'False_Negatives': false_negatives
})

print("\n" + "="*70)
print("MODEL PERFORMANCE AT DIFFERENT CONFIDENCE THRESHOLDS")
print("="*70)
print(df.to_string(index=False))

# ============================================================================
# PLOT 1: PR Curve with OTIF Overlay
# ============================================================================

fig, ax1 = plt.subplots(figsize=(14, 8))

# Primary y-axis: Precision and Recall
ax1.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Precision / Recall / F1 Score', fontsize=14, fontweight='bold', color='black')
ax1.set_ylim(0.3, 1.0)

# Plot Precision, Recall, and F1
line1 = ax1.plot(thresholds, precision, 'o-', linewidth=2.5, markersize=8, 
                 label='Precision', color='#2E86AB', alpha=0.8)
line2 = ax1.plot(thresholds, recall, 's-', linewidth=2.5, markersize=8, 
                 label='Recall', color='#A23B72', alpha=0.8)
line3 = ax1.plot(thresholds, f1_score, '^-', linewidth=2.5, markersize=8, 
                 label='F1 Score', color='#F18F01', alpha=0.8)

ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
ax1.tick_params(axis='x', labelsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# Secondary y-axis: OTIF Score
ax2 = ax1.twinx()
ax2.set_ylabel('OTIF Score', fontsize=14, fontweight='bold', color='#06A77D')
ax2.set_ylim(0.65, 1.0)

# Plot OTIF
line4 = ax2.plot(thresholds, otif_score, 'D-', linewidth=3, markersize=10, 
                 label='OTIF Score', color='#06A77D', alpha=0.9)
ax2.tick_params(axis='y', labelcolor='#06A77D', labelsize=11)

# Find optimal threshold (minimum OTIF threshold of 90%)
min_acceptable_otif = 0.90
optimal_idx = np.where(otif_score >= min_acceptable_otif)[0][0]
optimal_threshold = thresholds[optimal_idx]
optimal_precision = precision[optimal_idx]
optimal_recall = recall[optimal_idx]
optimal_f1 = f1_score[optimal_idx]
optimal_otif = otif_score[optimal_idx]

# Mark optimal threshold
ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.text(optimal_threshold + 1, 0.95, f'Optimal\nThreshold\n{optimal_threshold}%', 
         fontsize=11, color='red', fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

# Add shaded region for acceptable OTIF
ax2.axhspan(min_acceptable_otif, 1.0, alpha=0.1, color='green', label=f'Target OTIF ≥{min_acceptable_otif}')

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', fontsize=12, framealpha=0.9)

# Title
plt.title('SKU Availability Forecasting: Precision-Recall vs OTIF Score\nby Confidence Threshold', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('sku_pr_otif_curve.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot 1 saved as 'sku_pr_otif_curve.png'")

# ============================================================================
# PLOT 2: Detailed Threshold Analysis
# ============================================================================

key_thresholds = [50, 70, optimal_threshold, 90, 100]
key_indices = [np.where(thresholds == t)[0][0] for t in key_thresholds]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Precision comparison
axes[0, 0].bar(key_thresholds, precision[key_indices], color='#2E86AB', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Precision by Threshold', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Threshold (%)', fontsize=12)
axes[0, 0].set_ylabel('Precision', fontsize=12)
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].axhline(y=optimal_precision, color='red', linestyle='--', label=f'Optimal: {optimal_precision:.2%}')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Recall comparison
axes[0, 1].bar(key_thresholds, recall[key_indices], color='#A23B72', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Recall by Threshold', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Threshold (%)', fontsize=12)
axes[0, 1].set_ylabel('Recall', fontsize=12)
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].axhline(y=optimal_recall, color='red', linestyle='--', label=f'Optimal: {optimal_recall:.2%}')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# OTIF comparison
axes[1, 0].bar(key_thresholds, otif_score[key_indices], color='#06A77D', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('OTIF Score by Threshold', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Threshold (%)', fontsize=12)
axes[1, 0].set_ylabel('OTIF Score', fontsize=12)
axes[1, 0].set_ylim(0.65, 1.0)
axes[1, 0].axhline(y=min_acceptable_otif, color='orange', linestyle='--', linewidth=2, label=f'Target: {min_acceptable_otif:.0%}')
axes[1, 0].axhline(y=optimal_otif, color='red', linestyle='--', label=f'Optimal: {optimal_otif:.2%}')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# False Positives vs False Negatives
x_pos = np.arange(len(key_thresholds))
width = 0.35
axes[1, 1].bar(x_pos - width/2, false_positives[key_indices], width, label='False Positives', 
               color='#E63946', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x_pos + width/2, false_negatives[key_indices], width, label='False Negatives', 
               color='#457B9D', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('False Positives vs False Negatives', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Threshold (%)', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(key_thresholds)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Comprehensive Threshold Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('sku_threshold_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Plot 2 saved as 'sku_threshold_comparison.png'")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("OPTIMAL THRESHOLD ANALYSIS")
print("="*70)
print(f"Minimum Acceptable OTIF: {min_acceptable_otif:.1%}")
print(f"\n✅ Optimal Confidence Threshold: {optimal_threshold}%")
print(f"  ├─ Precision:  {optimal_precision:.2%}")
print(f"  ├─ Recall:     {optimal_recall:.2%}")
print(f"  ├─ F1 Score:   {optimal_f1:.2%}")
print(f"  └─ OTIF Score: {optimal_otif:.2%}")
print(f"\nFalse Positives at this threshold: {df.iloc[optimal_idx]['False_Positives']}")
print(f"False Negatives at this threshold: {df.iloc[optimal_idx]['False_Negatives']}")
print("="*70)

print("\n" + "="*70)
print("BUSINESS IMPACT SUMMARY")
print("="*70)
print("""
Key Insights:

1. LOWER THRESHOLDS (50-60%):
   • High Recall: Catch most available SKUs
   • Low Precision: Many false alarms (predict available when not)
   • ⚠️  Low OTIF: False positives hurt delivery performance

2. OPTIMAL THRESHOLD (70-75%):
   • Balanced Precision and Recall
   • ✅ Meets minimum OTIF target
   • Best trade-off for business operations

3. HIGHER THRESHOLDS (85-100%):
   • Very High Precision: Rarely wrong when predicting available
   • Low Recall: Miss many actually available SKUs
   • High OTIF but poor service (too conservative)

RECOMMENDATION: 
Use the threshold where OTIF first exceeds your minimum target 
while maintaining reasonable recall for customer service.
""")
print("="*70)

# Export results
df.to_csv('sku_threshold_analysis.csv', index=False)
print("\n✅ Results exported to 'sku_threshold_analysis.csv'")
print(f"\n🎯 FINAL RECOMMENDATION: Use {optimal_threshold}% confidence threshold")
print("="*70)

plt.show()

