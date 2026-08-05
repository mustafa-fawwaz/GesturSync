import matplotlib.pyplot as plt

# --- YOUR REAL RESULTS ---
accuracy = 98.80  
precision = 98.86
recall = 98.80
# -------------------------

labels = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]
colors = ['#4CAF50', '#2196F3', '#FFC107']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors, width=0.5)

# Add the percentage numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 110)
plt.ylabel('Percentage (%)', fontweight='bold')
plt.title('SVM Model Evaluation Metrics', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the image automatically
plt.savefig('accuracy_graph.png', bbox_inches='tight', dpi=300)
print("Graph saved successfully as 'accuracy_graph.png'!")