"""
Create visual architecture diagrams for the neural network models used in the research.
This script generates diagrams for Small CNN, ResNet50, and Vision Transformer (ViT)
architectures with both single-neuron and dual-neuron output configurations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import torch
from models import SmallCNN, ResNetModel, ViTModel

# Create output directory
os.makedirs('paper/figures', exist_ok=True)

# Define colors for different layer types with improved contrast and visual appeal
COLORS = {
    'input': '#D6EAF8',      # Light blue for input layers
    'conv': '#AED6F1',       # Medium blue for convolutional layers
    'bn': '#85C1E9',         # Darker blue for batch normalization
    'pool': '#5DADE2',       # Deep blue for pooling layers
    'dropout': '#3498DB',    # Royal blue for dropout layers
    'fc': '#2E86C1',         # Navy blue for fully connected layers
    'output': '#1B4F72',     # Dark blue for output layers
    'attention': '#F5B7B1',  # Light red for attention mechanisms
    'norm': '#F1948A',       # Medium red for normalization layers
    'mlp': '#EC7063',        # Deep red for MLP layers
    'residual': '#CB4335',   # Dark red for residual connections
    'frozen': '#D2B4DE',     # Light purple for frozen layers
    'unfrozen': '#AF7AC5',   # Darker purple for unfrozen layers
    'note': '#FCF3CF',       # Light yellow for notes/annotations
    'background': '#F5EEF8'  # Very light purple for backgrounds
}

def draw_box(ax, x, y, width, height, color, label=None, fontsize=8, alpha=0.7, edgecolor='black', linewidth=1, zorder=1):
    """Draw a box representing a layer with enhanced visual properties"""
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor=edgecolor, 
                    alpha=alpha, linewidth=linewidth, zorder=zorder)
    ax.add_patch(rect)
    if label:
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
                fontsize=fontsize, zorder=zorder+1)
    return rect

def draw_arrow(ax, start, end, color='black', style='simple', linewidth=1, zorder=2):
    """Draw an arrow between layers with enhanced visual properties"""
    if style == 'simple':
        arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color, 
                               mutation_scale=15, linewidth=linewidth, zorder=zorder)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color, 
                               connectionstyle=style, mutation_scale=15, 
                               linewidth=linewidth, zorder=zorder)
    ax.add_patch(arrow)
    return arrow

def create_small_cnn_diagram(output_neurons=1):
    """Create an enhanced diagram for the Small CNN architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the plot with more space
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title with more descriptive information
    output_type = "Single-Neuron" if output_neurons == 1 else "Dual-Neuron"
    ax.set_title(f'Small CNN Architecture with {output_type} Output Layer', fontsize=16, fontweight='bold')
    
    # Add subtitle with parameter information
    if output_neurons == 1:
        param_info = "~122K parameters (Single-Neuron Output)"
    else:
        param_info = "~122K parameters (Dual-Neuron Output)"
    ax.text(7, 7.5, param_info, fontsize=12, ha='center', va='center', style='italic')
    
    # Input layer with more details
    draw_box(ax, 0.5, 4, 1.2, 1.2, COLORS['input'], 'Input\n32×32×3\nRGB Image', fontsize=10, zorder=3)
    
    # First convolutional block - more detailed
    block1_y = 4
    block1_bg_x = 2.0
    block1_bg_width = 5.0
    block1_bg_height = 1.4
    
    # Background for block 1
    draw_box(ax, block1_bg_x, block1_y-0.1, block1_bg_width, block1_bg_height, 
             '#E8F8F5', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#A3E4D7')
    ax.text(block1_bg_x + 0.2, block1_y + 1.1, "Block 1: 32 Filters", fontsize=9, ha='left', va='center', style='italic')
    
    # First conv block components
    draw_box(ax, 2.2, block1_y, 0.9, 1.2, COLORS['conv'], 'Conv2D\n3×3, 32\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 3.3, block1_y, 0.7, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 4.2, block1_y, 0.9, 1.2, COLORS['conv'], 'Conv2D\n3×3, 32\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 5.3, block1_y, 0.7, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 6.2, block1_y, 0.9, 1.2, COLORS['pool'], 'MaxPool\n2×2\n16×16', fontsize=9, zorder=3)
    draw_box(ax, 7.3, block1_y, 0.8, 1.2, COLORS['dropout'], 'Dropout\n0.2', fontsize=9, zorder=3)
    
    # Second convolutional block
    block2_y = 4
    block2_bg_x = 8.4
    block2_bg_width = 5.0
    block2_bg_height = 1.4
    
    # Background for block 2
    draw_box(ax, block2_bg_x, block2_y-0.1, block2_bg_width, block2_bg_height, 
             '#EAF2F8', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#AED6F1')
    ax.text(block2_bg_x + 0.2, block2_y + 1.1, "Block 2: 64 Filters", fontsize=9, ha='left', va='center', style='italic')
    
    # Second conv block components
    draw_box(ax, 8.6, block2_y, 0.9, 1.2, COLORS['conv'], 'Conv2D\n3×3, 64\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 9.7, block2_y, 0.7, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 10.6, block2_y, 0.9, 1.2, COLORS['conv'], 'Conv2D\n3×3, 64\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 11.7, block2_y, 0.7, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 12.6, block2_y, 0.9, 1.2, COLORS['pool'], 'MaxPool\n2×2\n8×8', fontsize=9, zorder=3)
    
    # Third convolutional block (shown as a note)
    add_info_box(ax, 5, 6, "Block 3: 128 Filters\n- Conv2D 3×3, 128, ReLU\n- Batch Normalization\n- Conv2D 3×3, 128, ReLU\n- Batch Normalization\n- MaxPool 2×2 → 4×4\n- Dropout 0.3", 
                fontsize=9, color=COLORS['note'])
    
    # Add dimension information (moved to top of diagram)
    add_info_box(ax, 9, 6, "Feature Dimensions:\n- Input: 32×32×3\n- After Block 1: 16×16×32\n- After Block 2: 8×8×64\n- After Block 3: 4×4×128\n- Flattened: 2048 features", 
                fontsize=9, color=COLORS['note'])
    
    # Fully connected layers
    fc_y = 2
    fc_bg_x = 2.0
    fc_bg_width = 10.5
    fc_bg_height = 1.4
    
    # Background for FC layers
    draw_box(ax, fc_bg_x, fc_y-0.1, fc_bg_width, fc_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    ax.text(fc_bg_x + 0.2, fc_y + 1.1, "Classifier", fontsize=9, ha='left', va='center', style='italic')
    
    # Output layer with more details (now appears first, at x=2.5)
    if output_neurons == 1:
        draw_box(ax, 2.5, fc_y, 1.2, 1.2, COLORS['output'], 'FC 1\nSigmoid\nBinary\nOutput', fontsize=9, zorder=3)
    else:
        draw_box(ax, 2.5, fc_y, 1.2, 1.2, COLORS['output'], 'FC 2\nSoftmax\nBinary\nOutput', fontsize=9, zorder=3)

    # FC layers with more details - reversed
    draw_box(ax, 4.0, fc_y, 0.8, 1.2, COLORS['dropout'], 'Dropout\n0.5', fontsize=9, zorder=3)
    draw_box(ax, 5.0, fc_y, 0.8, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 6.1, fc_y, 1.8, 1.2, COLORS['fc'], 'Flatten\n→ FC 256\nReLU', fontsize=9, zorder=3)

    # Feature dimensions information moved to top of diagram
    
    # Arrows connecting components - Block 1
    draw_arrow(ax, (1.7, 4.6), (2.2, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (3.1, 4.6), (3.3, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (4.0, 4.6), (4.2, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (5.1, 4.6), (5.3, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (6.0, 4.6), (6.2, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (7.1, 4.6), (7.3, 4.6), linewidth=1.5, zorder=2)
    
    # Connect Block 1 to Block 2
    draw_arrow(ax, (8.1, 4.6), (8.6, 4.6), linewidth=1.5, zorder=2)
    
    # Arrows connecting components - Block 2
    draw_arrow(ax, (9.5, 4.6), (9.7, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (10.4, 4.6), (10.6, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (11.5, 4.6), (11.7, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (12.4, 4.6), (12.6, 4.6), linewidth=1.5, zorder=2)
    
    # Connect Block 3 (imaginary) to FC layer with proper flow direction
    draw_arrow(ax, (13.5, 4.6), (13.7, 4.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (13.7, 4.6), (13.7, 2.6), style='arc3,rad=-0.3', linewidth=1.5, zorder=2)
    draw_arrow(ax, (13.7, 2.6), (2.5, 2.6), style='arc3,rad=0', linewidth=1.5, zorder=2)
    
    # Connect FC components with proper flow direction
    draw_arrow(ax, (4.3, 2.6), (4.6, 2.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (5.4, 2.6), (5.7, 2.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (6.5, 2.6), (6.8, 2.6), linewidth=1.5, zorder=2)
    
    # Add legend with more details
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input'], edgecolor='black', alpha=0.7, label='Input Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['conv'], edgecolor='black', alpha=0.7, label='Convolutional Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['bn'], edgecolor='black', alpha=0.7, label='Batch Normalization'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['pool'], edgecolor='black', alpha=0.7, label='Pooling Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['dropout'], edgecolor='black', alpha=0.7, label='Dropout'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['fc'], edgecolor='black', alpha=0.7, label='Fully Connected Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output'], edgecolor='black', alpha=0.7, label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              ncol=4, fontsize=9)
    
    plt.tight_layout()
    
    # Save the figure with higher resolution
    output_file = f'paper/figures/small_cnn_{output_neurons}neuron_architecture.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Small CNN diagram to {output_file}")
    plt.close()

def create_resnet_diagram(output_neurons=1):
    """Create an enhanced diagram for the ResNet50 architecture"""
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Set up the plot with more space
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title with more descriptive information
    output_type = "Single-Neuron" if output_neurons == 1 else "Dual-Neuron"
    ax.set_title(f'ResNet50 Architecture with {output_type} Output Layer', fontsize=16, fontweight='bold')
    
    # Add subtitle with parameter information
    if output_neurons == 1:
        param_info = "~23.5M parameters (Single-Neuron Output)"
    else:
        param_info = "~23.5M parameters (Dual-Neuron Output)"
    ax.text(7, 8.5, param_info, fontsize=12, ha='center', va='center', style='italic')
    
    # Input layer with more details
    draw_box(ax, 0.5, 5, 1.2, 1.2, COLORS['input'], 'Input\n224×224×3\nRGB Image', fontsize=10, zorder=3)
    
    # Initial layers - stem
    stem_bg_x = 2.0
    stem_bg_width = 3.5
    stem_bg_height = 1.4
    
    # Background for stem
    draw_box(ax, stem_bg_x, 5-0.1, stem_bg_width, stem_bg_height, 
             '#E8F8F5', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#A3E4D7')
    ax.text(stem_bg_x + 0.2, 5 + 1.1, "Stem", fontsize=9, ha='left', va='center', style='italic')
    
    # Stem components
    draw_box(ax, 2.2, 5, 1.0, 1.2, COLORS['conv'], 'Conv2D\n7×7, 64\nstride 2', fontsize=9, zorder=3)
    draw_box(ax, 3.4, 5, 0.8, 1.2, COLORS['bn'], 'Batch\nNorm\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 4.4, 5, 1.0, 1.2, COLORS['pool'], 'MaxPool\n3×3\nstride 2', fontsize=9, zorder=3)
    
    # ResNet blocks with detailed structure
    # Layer 1 - 3 blocks
    layer1_bg_x = 5.8
    layer1_bg_width = 1.5
    layer1_bg_height = 1.4
    
    # Background for layer 1
    draw_box(ax, layer1_bg_x, 5-0.1, layer1_bg_width, layer1_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    
    # Layer 1 block
    draw_box(ax, 6.0, 5, 1.1, 1.2, COLORS['residual'], 'Layer 1\n3 Blocks\n64 channels', fontsize=9, zorder=3)
    
    # Layer 2 - 4 blocks
    layer2_bg_x = 7.6
    layer2_bg_width = 1.5
    layer2_bg_height = 1.4
    
    # Background for layer 2
    draw_box(ax, layer2_bg_x, 5-0.1, layer2_bg_width, layer2_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    
    # Layer 2 block
    draw_box(ax, 7.8, 5, 1.1, 1.2, COLORS['residual'], 'Layer 2\n4 Blocks\n128 channels', fontsize=9, zorder=3)
    
    # Layer 3 - 6 blocks
    layer3_bg_x = 9.4
    layer3_bg_width = 1.5
    layer3_bg_height = 1.4
    
    # Background for layer 3
    draw_box(ax, layer3_bg_x, 5-0.1, layer3_bg_width, layer3_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    
    # Layer 3 block
    draw_box(ax, 9.6, 5, 1.1, 1.2, COLORS['residual'], 'Layer 3\n6 Blocks\n256 channels', fontsize=9, zorder=3)
    
    # Layer 4 - 3 blocks
    layer4_bg_x = 11.2
    layer4_bg_width = 1.5
    layer4_bg_height = 1.4
    
    # Background for layer 4
    draw_box(ax, layer4_bg_x, 5-0.1, layer4_bg_width, layer4_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    
    # Layer 4 block
    draw_box(ax, 11.4, 5, 1.1, 1.2, COLORS['residual'], 'Layer 4\n3 Blocks\n512 channels', fontsize=9, zorder=3)
    
    # Add residual block detail
    add_info_box(ax, 3.5, 7.5, "Bottleneck Block Structure:\n1. 1×1 Conv (reduce channels)\n2. 3×3 Conv\n3. 1×1 Conv (increase channels)\n4. Add input (residual connection)\n5. ReLU activation", 
                fontsize=9, color=COLORS['note'])
    
    # Add residual block description (moved to top between other info boxes)
    add_info_box(ax, 7.0, 7.5, "Each residual block contains:\n1×1 conv → 3×3 conv → 1×1 conv\nwith batch normalization and ReLU", 
                fontsize=9, color=COLORS['note'])
    
    # Add feature map sizes
    add_info_box(ax, 10.5, 7.5, "Feature Map Dimensions:\n- Input: 224×224×3\n- After Stem: 56×56×64\n- After Layer 1: 56×56×256\n- After Layer 2: 28×28×512\n- After Layer 3: 14×14×1024\n- After Layer 4: 7×7×2048", 
                fontsize=9, color=COLORS['note'])
    
    # Global average pooling and custom classifier
    classifier_bg_x = 2.0
    classifier_bg_width = 10.5
    classifier_bg_height = 1.4
    classifier_y = 2.5
    
    # Background for classifier
    draw_box(ax, classifier_bg_x, classifier_y-0.1, classifier_bg_width, classifier_bg_height, 
             '#F5EEF8', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#D2B4DE')
    ax.text(classifier_bg_x + 0.2, classifier_y + 1.1, "Custom Classifier (Replacing Original FC-1000)", 
            fontsize=9, ha='left', va='center', style='italic')
    
    # Output layer with more details (now appears first, but moved to x=2.5)
    if output_neurons == 1:
        draw_box(ax, 2.5, classifier_y, 1.2, 1.2, COLORS['output'], 'FC 1\nSigmoid\nBinary\nOutput', fontsize=9, zorder=3)
    else:
        draw_box(ax, 2.5, classifier_y, 1.2, 1.2, COLORS['output'], 'FC 2\nSoftmax\nBinary\nOutput', fontsize=9, zorder=3)

    # Pooling and FC layers (fully reversed)
    draw_box(ax, 4.2, classifier_y, 1.2, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 5.9, classifier_y, 1.2, 1.2, COLORS['fc'], 'FC 128\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 7.6, classifier_y, 1.2, 1.2, COLORS['bn'], 'Batch\nNorm', fontsize=9, zorder=3)
    draw_box(ax, 9.3, classifier_y, 1.2, 1.2, COLORS['fc'], 'FC 512\nReLU', fontsize=9, zorder=3)
    draw_box(ax, 11.0, classifier_y, 1.2, 1.2, COLORS['pool'], 'Global\nAvg Pool\n2048', fontsize=9, zorder=3)

    # Arrows connecting components - Main flow with thicker lines
    draw_arrow(ax, (1.7, 5.6), (2.2, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (3.2, 5.6), (3.4, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (4.2, 5.6), (4.4, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (5.4, 5.6), (6.0, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (7.1, 5.6), (7.8, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (8.9, 5.6), (9.6, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (10.7, 5.6), (11.4, 5.6), linewidth=1.5, zorder=2)
    
    # Connect to global average pooling with clearer path and proper flow direction
    draw_arrow(ax, (12.5, 5.6), (12.8, 5.6), linewidth=1.5, zorder=2)
    draw_arrow(ax, (12.8, 5.6), (12.8, 3.1), style='arc3,rad=-0.3', linewidth=1.5, zorder=2)
    draw_arrow(ax, (12.8, 3.1), (2.5, 3.1), style='arc3,rad=0', linewidth=1.5, zorder=2)
    
    # Connect classifier components with consistent styling and proper flow direction
    draw_arrow(ax, (3.7, 3.1), (4.2, 3.1), linewidth=1.5, zorder=2)
    draw_arrow(ax, (5.4, 3.1), (5.9, 3.1), linewidth=1.5, zorder=2)
    draw_arrow(ax, (7.1, 3.1), (7.6, 3.1), linewidth=1.5, zorder=2)
    draw_arrow(ax, (8.8, 3.1), (9.3, 3.1), linewidth=1.5, zorder=2)
    draw_arrow(ax, (10.5, 3.1), (11.0, 3.1), linewidth=1.5, zorder=2)
    
    # Add residual connections (simplified)
    # For Layer 1
    draw_arrow(ax, (5.2, 3.2), (5.4, 3), color='red', style='arc3,rad=0.3')
    draw_arrow(ax, (6.2, 3), (6.4, 3.2), color='red', style='arc3,rad=0.3')
    
    # For Layer 2
    draw_arrow(ax, (6.8, 3.2), (7, 3), color='red', style='arc3,rad=0.3')
    draw_arrow(ax, (7.8, 3), (8, 3.2), color='red', style='arc3,rad=0.3')
    
    # For Layer 3
    draw_arrow(ax, (8.4, 3.2), (8.6, 3), color='red', style='arc3,rad=0.3')
    draw_arrow(ax, (9.4, 3), (9.6, 3.2), color='red', style='arc3,rad=0.3')
    
    # For Layer 4
    draw_arrow(ax, (10, 3.2), (10.2, 3), color='red', style='arc3,rad=0.3')
    draw_arrow(ax, (11, 3), (11.2, 3.2), color='red', style='arc3,rad=0.3')
    
    # Removed the overlapping note about residual blocks (moved to top of diagram)
    
    # Add a note about transfer learning
    # ax.text(3, 6.5, "Transfer Learning Strategy:\nFreeze all layers except Layer 4 and classifier", 
    #         ha='center', va='center', fontsize=9, style='italic',
    #         bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input'], edgecolor='black', alpha=0.7, label='Input Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['conv'], edgecolor='black', alpha=0.7, label='Convolutional Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['bn'], edgecolor='black', alpha=0.7, label='Batch Normalization'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['pool'], edgecolor='black', alpha=0.7, label='Pooling Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['residual'], edgecolor='black', alpha=0.7, label='Residual Block'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['fc'], edgecolor='black', alpha=0.7, label='Fully Connected Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output'], edgecolor='black', alpha=0.7, label='Output Layer'),
        Line2D([0], [0], color='red', lw=2, label='Residual Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              ncol=4, fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = f'paper/figures/resnet50_{output_neurons}neuron_architecture.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved ResNet50 diagram to {output_file}")
    plt.close()

def create_vit_diagram(output_neurons=1):
    """Create an enhanced diagram for the Vision Transformer architecture"""
    # Create a much larger figure for the ViT architecture with clear separation
    fig, ax = plt.subplots(figsize=(16, 16))  # Drastically increased height
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)  # Drastically increased ylim
    ax.axis('off')
    
    # Title with more descriptive information
    if output_neurons == 1:
        title = "Vision Transformer (ViT) Architecture with Single-Neuron Output"
        param_info = "~86M parameters (Single-Neuron Output)       ViT-B/16"
    else:
        title = "Vision Transformer (ViT) Architecture with Dual-Neuron Output"
        param_info = "~86M parameters (Dual-Neuron Output)       ViT-B/16"
    ax.text(8, 15, title, fontsize=14, weight='bold', ha='center', va='center')
    ax.text(8, 14.2, param_info, fontsize=12, ha='center', va='center', style='italic')
    
    # Transformer Encoder section - moved up and stacked vertically
    encoder_bg_x = 7.0
    encoder_bg_width = 3.0
    encoder_bg_height = 6.0
    encoder_y = 7.0  # Adjusted for vertical stack
    
    # Background for transformer encoder
    draw_box(ax, encoder_bg_x, encoder_y-0.1, encoder_bg_width, encoder_bg_height, 
             '#E8F8F5', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#82E0AA')
    ax.text(encoder_bg_x + 0.2, encoder_y + 5.7, "Transformer Encoder", 
            fontsize=10, ha='left', va='center', style='italic')
    
    # Transformer encoder blocks - stacked vertically
    block_width = 2.5
    block_height = 0.4
    block_spacing = 0.1
    block_x = encoder_bg_x + 0.25
    
    # Frozen blocks (1-10)
    for i in range(10):
        block_y = encoder_y + 5.0 - (i * (block_height + block_spacing))
        draw_box(ax, block_x, block_y, block_width, block_height, 
                COLORS['frozen'], f'Block {i+1}', fontsize=9, zorder=3)
    
    # Unfrozen blocks (11-12)
    for i in range(10, 12):
        block_y = encoder_y + 5.0 - (i * (block_height + block_spacing))
        draw_box(ax, block_x, block_y, block_width, block_height, 
                COLORS['unfrozen'], f'Block {i+1}', fontsize=9, zorder=3)
    
    # Input and patch embedding section - moved up
    # Input image
    draw_box(ax, 1.0, 11.0, 1.0, 1.2, COLORS['input'], 'Input\n224×224×3\nRGB Image', fontsize=10, zorder=3)
    
    # Patch embedding process
    draw_box(ax, 2.7, 11.0, 1.2, 1.2, COLORS['conv'], 'Patch\nEmbedding', fontsize=10, zorder=3)
    draw_box(ax, 4.1, 11.0, 0.8, 1.2, COLORS['conv'], 'Class\nToken', fontsize=10, zorder=3)
    draw_box(ax, 5.1, 11.0, 1.0, 1.2, COLORS['conv'], 'Position\nEmbedding', fontsize=10, zorder=3)
    
    # MLP Head section - positioned between encoder and classifier
    head_bg_x = 11.5
    head_bg_width = 1.5
    head_bg_height = 1.6
    head_y = 6.0  # Positioned in the middle
    
    # Background for MLP head
    draw_box(ax, head_bg_x, head_y-0.1, head_bg_width, head_bg_height, 
             '#EBF5FB', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#85C1E9')
    ax.text(head_bg_x + 0.2, head_y + 1.3, "Head", fontsize=10, ha='left', va='center', style='italic')
    
    # MLP Head component
    draw_box(ax, 12.2, head_y, 1.1, 1.4, COLORS['fc'], 'MLP\nHead\n(768→768)', fontsize=10, zorder=3)
    
    # Custom classifier section - moved to bottom of diagram
    classifier_bg_x = 7.0
    classifier_bg_width = 6.0
    classifier_bg_height = 1.6
    classifier_y = 2.5  # Positioned much lower in the diagram
    
    # Background for custom classifier
    draw_box(ax, classifier_bg_x, classifier_y-0.1, classifier_bg_width, classifier_bg_height, 
             '#F5EEF8', None, alpha=0.3, zorder=1, linewidth=1.5, edgecolor='#D2B4DE')
    ax.text(classifier_bg_x + 0.2, classifier_y + 1.3, "Classifier", 
            fontsize=10, ha='left', va='center', style='italic')
    
    # Output layer (moved to the beginning)
    if output_neurons == 1:
        draw_box(ax, 7.3, classifier_y, 1.1, 1.4, COLORS['output'], 'FC 1\nSigmoid\nBinary\nOutput', fontsize=10, zorder=3)
    else:
        draw_box(ax, 7.3, classifier_y, 1.1, 1.4, COLORS['output'], 'FC 2\nSoftmax\nBinary\nOutput', fontsize=10, zorder=3)

    # Reversed custom classifier components
    draw_box(ax, 8.4, classifier_y, 0.9, 1.4, COLORS['bn'], 'Batch\nNorm', fontsize=10, zorder=3)
    draw_box(ax, 9.5, classifier_y, 1.1, 1.4, COLORS['fc'], 'FC 128\nReLU', fontsize=10, zorder=3)
    draw_box(ax, 10.6, classifier_y, 0.9, 1.4, COLORS['bn'], 'Batch\nNorm', fontsize=10, zorder=3)
    draw_box(ax, 11.9, classifier_y, 1.1, 1.4, COLORS['fc'], 'FC 256\nReLU', fontsize=10, zorder=3)

    # Arrows connecting components - completely redone for new layout
    # Input to tokenization
    draw_arrow(ax, (2.0, 11.6), (2.7, 11.6), linewidth=1.8, zorder=2)
    
    # Connect tokenization components
    draw_arrow(ax, (3.9, 11.6), (4.1, 11.6), linewidth=1.8, zorder=2)
    draw_arrow(ax, (4.9, 11.6), (5.1, 11.6), linewidth=1.8, zorder=2)
    
    # Connect to transformer encoder
    draw_arrow(ax, (6.1, 11.6), (6.5, 11.6), linewidth=1.8, zorder=2)
    draw_arrow(ax, (6.5, 11.6), (6.5, 12.5), linewidth=1.8, zorder=2)
    draw_arrow(ax, (6.5, 12.5), (7.0, 12.5), linewidth=1.8, zorder=2)
    
    # Connect from transformer to MLP head
    draw_arrow(ax, (10.0, 10.0), (11.5, 6.6), style='arc3,rad=-0.3', linewidth=1.8, zorder=2)
    
    # Connect MLP head to custom classifier
    draw_arrow(ax, (13.3, 6.6), (14.0, 6.6), linewidth=1.8, zorder=2)
    draw_arrow(ax, (14.0, 6.6), (14.0, 3.3), style='arc3,rad=-0.3', linewidth=1.8, zorder=2)
    draw_arrow(ax, (14.0, 3.3), (7.3, 3.3), style='arc3,rad=0', linewidth=1.8, zorder=2)
    
    # Connect classifier components
    draw_arrow(ax, (8.4, 3.3), (8.6, 3.3), linewidth=1.8, zorder=2)
    draw_arrow(ax, (9.5, 3.3), (9.7, 3.3), linewidth=1.8, zorder=2)
    draw_arrow(ax, (10.8, 3.3), (11.0, 3.3), linewidth=1.8, zorder=2)
    draw_arrow(ax, (11.9, 3.3), (12.1, 3.3), linewidth=1.8, zorder=2)
    
    # Add details about transformer blocks - repositioned
    add_info_box(ax, 3.5, 8.5, "Each Transformer Encoder Block contains:\n- Multi-Head Self-Attention (12 heads)\n- Layer Normalization\n- MLP with GELU activation\n- Residual connections", 
                fontsize=10, color=COLORS['note'], zorder=3)
    
    # Add note about patch embedding - moved near other info boxes
    add_info_box(ax, 3.5, 6.5, "Patch Embedding Process:\n- Split 224×224 image into 16×16 patches\n- Flatten patches to 768-dim vectors\n- Add learnable class token [CLS]\n- Add positional embeddings", 
                fontsize=10, color=COLORS['note'], zorder=3)
    
    # Add note about model architecture - repositioned
    add_info_box(ax, 3.5, 4.5, "Vision Transformer (ViT) Architecture:\n- Based on ViT-B/16 model\n- 12 transformer encoder blocks\n- 768-dimensional embeddings\n- 12 attention heads per block\n- 86M total parameters", 
                fontsize=10, color=COLORS['note'], zorder=3)
    
    # Add note about transfer learning
    # add_info_box(ax, 3.5, 9.5, "Transfer Learning Strategy:\n- Freeze first 10 encoder blocks\n- Train only blocks 11-12 and classifier\n- Leverages ImageNet pre-trained weights", 
    #             fontsize=10, color=COLORS['note'], zorder=3)
    
    # Add legend - adjusted position for new figure size
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input'], edgecolor='black', alpha=0.7, label='Input/Embedding'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['conv'], edgecolor='black', alpha=0.7, label='Patch Embedding'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['frozen'], edgecolor='black', alpha=0.7, label='Frozen Encoder Block'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['unfrozen'], edgecolor='black', alpha=0.7, label='Unfrozen Encoder Block'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['fc'], edgecolor='black', alpha=0.7, label='Fully Connected Layer'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['bn'], edgecolor='black', alpha=0.7, label='Batch Normalization'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output'], edgecolor='black', alpha=0.7, label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=4, fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure with higher resolution
    output_file = f'paper/figures/vit_{output_neurons}neuron_architecture.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved ViT diagram to {output_file}")
    plt.close()
def add_info_box(ax, x, y, text, fontsize=9, alpha=0.7, color='#FCF3CF', zorder=3):
    """Add an information box with descriptive text"""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, style='italic',
            bbox=dict(facecolor=color, alpha=alpha, boxstyle='round,pad=0.5', zorder=zorder))

def create_model_architecture_diagrams():
    """Create architecture diagrams for all models with both output configurations"""
    print("Generating architecture diagrams...")
    
    # Small CNN
    create_small_cnn_diagram(output_neurons=1)
    create_small_cnn_diagram(output_neurons=2)
    
    # ResNet50
    create_resnet_diagram(output_neurons=1)
    create_resnet_diagram(output_neurons=2)
    
    # Vision Transformer
    create_vit_diagram(output_neurons=1)
    create_vit_diagram(output_neurons=2)
    
    print("All architecture diagrams generated successfully!")

if __name__ == "__main__":
    create_model_architecture_diagrams()
