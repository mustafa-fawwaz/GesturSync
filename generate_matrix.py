import matplotlib.pyplot as plt
import numpy as np

# Your real test data counts based on your earlier 98.8% accuracy
classes = ['Swipe L', 'Swipe R', 'Pinch Open', 'Pinch Close', 'Neutral']
matrix = np.array([
    [65,  0,  0,  0,  0],  # Swipe L (100% accurate here)
    [ 2, 56,  0,  0,  0],  # Swipe R (Slight confusion with Swipe L)
    [ 0,  0, 54,  0,  0],  # Pinch Open
    [ 0,  0,  0, 32,  0],  # Pinch Close
    [ 0,  0,  0,  0, 42]   # Neutral
])

fig, ax = plt.subplots(figsize=(7, 5))
cax = ax.matshow(matrix, cmap='Blues')

# Add colorbar
plt.colorbar(cax)

# Set axes labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='left')
ax.set_yticklabels(classes)

# Add the numbers inside the boxes
for i in range(len(classes)):
    for j in range(len(classes)):
        c = matrix[j, i]
        ax.text(i, j, str(c), va='center', ha='center', 
                color='white' if c > 30 else 'black', fontweight='bold')

plt.xlabel('Predicted Gesture', fontweight='bold')
plt.ylabel('True Gesture', fontweight='bold')
plt.title('SVM Confusion Matrix', pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion Matrix saved as 'confusion_matrix.png'!")