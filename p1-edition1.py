import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# دریافت تعداد قراردادها از کاربر
n = int(input("تعداد قراردادها را وارد کنید: "))
p = 0.1  # احتمال آنلاین شدن هر فرد

# محاسبه توزیع دوجمله‌ای
k = np.arange(0, n+1)
probabilities = binom.pmf(k, n, p)

# پیدا کردن k با بالاترین احتمال
most_probable_k = k[np.argmax(probabilities)]
print(f"بالاترین احتمال برای k: {most_probable_k}")

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.plot(k, probabilities, label='Binomial distribution')
plt.xlabel('Number of people online')
plt.ylabel('possibility')
plt.title('Binomial distribution for the number of people online')
plt.legend()
plt.grid(True)
plt.show()
