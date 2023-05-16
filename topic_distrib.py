import numpy as np
import matplotlib.pyplot as plt

all_scores = np.load("all_scores.npy1000.npy")

topic_counts = [0 for _ in range(19)]
labels = ['0: arts_&_culture', '1: business_&_entrepreneurs', '2: celebrity_&_pop_culture',
          '3: diaries_&_daily_life', '4: family', '5: fashion_&_style',
          '6: film_tv_&_video', '7: fitness_&_health', '8: food_&_dining',
          '9: gaming', '10: learning_&_educational', '11: music',
          '12: news_&_social_concern', '13: other_hobbies', '14: relationships',
          '15: science_&_technology', '16: sports', '17: travel_&_adventure',
          '18: youth_&_student_life']
for raw_score in all_scores:
    score = np.exp(raw_score)/sum(np.exp(raw_score))
    topic_counts[np.argmax(score)] += 1
plt.bar(labels, topic_counts)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
