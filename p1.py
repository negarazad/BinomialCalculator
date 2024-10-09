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
#تنظیم کردن اندازه نمودار
plt.figure(figsize=(10, 6))
#رسم کردن نمودار توزیع دو جمله ای
plt.plot(k, probabilities, label='Binomial distribution')
#نامگذاری محور ها
plt.xlabel('Number of people online')
plt.ylabel('possibility')
#عنوان دهی به خود نمودار
plt.title('Binomial distribution for the number of people online')
#راهنمای نمودار
plt.legend()
#شبکه بندی کردن نمودار
plt.grid(True)
#نمایش دادن نمودار
plt.show()
