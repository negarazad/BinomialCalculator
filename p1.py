import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# پارامترها
n = 5000  # تعداد قراردادها
p = 0.1    # احتمال آنلاین شدن هر فرد

# محاسبه توزیع دوجمله‌ای
k = np.arange(0, n+1)
probabilities = binom.pmf(k, n, p)

# پیدا کردن k با بالاترین احتمال
most_probable_k = k[np.argmax(probabilities)]
print(f"بالاترین احتمال برای k: {most_probable_k}")

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.plot(k, probabilities, label='توزیع دوجمله‌ای')
plt.xlabel('تعداد افراد آنلاین')
plt.ylabel('احتمال')
plt.title('توزیع دوجمله‌ای برای تعداد افراد آنلاین')
plt.legend()
plt.grid(True)
plt.show()
