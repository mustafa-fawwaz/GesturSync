import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up a wide, blank canvas
fig, ax = plt.subplots(figsize=(14, 2.5))
ax.axis('off')

# Define the 5 stages of your pipeline to match your paper
labels = [
    "1. Video Ingestion\n(OpenCV Webcam)",
    "2. Feature Extraction\n(MediaPipe 3D)",
    "3. Classification\n(SVM Model)",
    "4. Heuristic Logic\n(Distance Mapping)",
    "5. System Execution\n(PyAutoGUI)"
]

box_w = 2.4  # Box width
box_h = 1.0  # Box height
gap = 0.8    # Space between boxes

for i, label in enumerate(labels):
    x = i * (box_w + gap)
    y = 0
    
    # Draw a professional-looking styled box
    rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.1", 
                                  edgecolor='#1976D2', facecolor='#E3F2FD', lw=2)
    ax.add_patch(rect)
    
    # Add the perfectly spelled text inside
    ax.text(x + box_w/2, y + box_h/2, label, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#0D47A1')
    
    # Draw connecting arrows between the boxes
    if i < len(labels) - 1:
        arrow_start = x + box_w
        arrow_end = x + box_w + gap
        ax.annotate('', xy=(arrow_end, y + box_h/2), xytext=(arrow_start, y + box_h/2),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color='#424242'))

# Adjust limits and save
plt.xlim(-0.2, len(labels) * (box_w + gap))
plt.ylim(-0.2, 1.2)
plt.tight_layout()

# Save it exactly as your LaTeX file expects
plt.savefig('architecture.png', dpi=300, bbox_inches='tight')
print("Architecture diagram generated and saved as 'architecture.png'!")