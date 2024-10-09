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

# محاسبه احتمال آنلاین شدن 1000 نفر
prob_1000 = binom.pmf(200, n, p)
print(f"احتمال آنلاین شدن همزمان 1000 نفر از 5000 نفر: {prob_1000}")

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.plot(k, probabilities, label='Binomial distribution')
plt.scatter([200], [prob_1000], color='red', zorder=5, label='Probability of 1000 people online')
plt.xlabel('Number of people online')
plt.ylabel('possibility')
plt.title('Binomial distribution for the number of people online')
plt.legend()
plt.grid(True)
plt.show()
