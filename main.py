import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ODOMETER = 'ODOMETER'
SPEED = 'SPEED'

data_frame = pd.read_csv("data.csv")

data_frame.describe()
df['reviews_per_month',
   'number_of_reviews',
   'price',
   'accommodates',
   'beds',
   'minimum_nights'
].append(bool_map(df['host_has_profile_pic'])).append(bool_map(df['host_identity_verified'])).append(
    bool_map(df['instant_bookable']))

# clustering = KMeans(n_clusters=6).fit(data_frame[[SPEED]])
# labels = clustering.labels_
#
# odometer_points = data_frame[[ODOMETER]]
# speed_points = data_frame[[SPEED]]
#
# plot = plt.scatter(x=odometer_points, y=speed_points, c=labels)
# plt.show()
#
# print(labels)
