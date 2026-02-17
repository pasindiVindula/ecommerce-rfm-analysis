# ============================================================
# TEST: Verify all libraries are installed correctly
# ============================================================

print("Testing all imports...")

import pandas as pd
print(f"✅ pandas {pd.__version__}")

import numpy as np
print(f"✅ numpy {np.__version__}")

import matplotlib
print(f"✅ matplotlib {matplotlib.__version__}")

import matplotlib.pyplot as plt

import seaborn as sns
print(f"✅ seaborn {sns.__version__}")

import sklearn
print(f"✅ scikit-learn {sklearn.__version__}")

import scipy
print(f"✅ scipy {scipy.__version__}")

print("\n🎉 All libraries installed! Ready to run analysis.")

# ============================================================
# QUICK TEST: Generate a simple plot
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [23, 45, 56, 78, 45, 90, 34, 67, 89, 12]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker='o', color='steelblue', linewidth=2)
plt.title('VS Code Setup Test - Plot Working!')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/test_plot.png')
plt.show()

print("✅ Plot saved to visualizations/test_plot.png")
print("\n🚀 VS Code is fully set up for your analysis!")