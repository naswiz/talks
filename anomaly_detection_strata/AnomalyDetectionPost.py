

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
plotBlue = sns.color_palette()[0]
np.random.seed(3)


# # Anomaly Detection & Probabilistic Programming
# 
# This post will present a short survey on popular methods in anomaly detection. After exploring some of the goals and limitations of these methods, we will suggest that probabilistic programming provides an easy way to formulate more robust anomaly detection models.

# ### What is Anomaly Detection?
# 
# Anomaly detection algorithms detect observations that are significantly different from most of what you've seen before.
# 
# One classic example here is in detecting credit card fraud: how do we automatically detect purchases that a legitimate credit card owner is very unlikely to have made?
# 
# Another is in systems security: how do we detect activity on a network that's unlikely to be caused be a legitimate user?
# 
# Anomaly detection is often done by building a probabilistic model of your data. This means you can see what the probability of observing every possible event is under your model. When you observe an event that has sufficiently low probability, you label it as anomalous.

# ### Challenges in Anomaly Detection
# 
# In this post, we'll explore some of the standard techniques for building these probabilistic models from observed data, starting from the simplest and increasing in complexity. We’ll find that some of the traditional approaches involve manually manipulating distributions until they ‘look’ right. We’ll also find that appropriately expressing the relationships between features quickly becomes difficult and inexact. Lastly, the traditional methods make it difficult to propagate uncertainty through multiple steps of our algorithm.
# 
# We'll conclude by looking at probabilistic programming languages that can help overcome each of those limitations with minimal friction.

# ### Normally Distributed Single Variable
# 
# In this section, we will see how anomaly detection can work for observations with a single feature.
# 
# The simplest possible case for anomaly detection is observational data with a single, normally distributed feature. We'll generate 1000 such observations.

N = 1000
X1 = np.random.normal(4, 12, N)

def plot_dist_single_val():

  f, axes = plt.subplots(nrows=2, sharex=True)
  axes[0].set_xlim(-50, 50)
  axes[0].scatter(X1, np.zeros(N), marker='x', c=plotBlue)
  axes[1].hist(X1, bins=50)
  plt.show()

plot_dist_single_val()
# To model this data as a normal distribution, we take the mean and the standard deviation from the sample we have.



sample_mean = X1.mean()
sample_sigma = X1.std()
print('Sample Mean:', sample_mean)
print('Sample Standard Deviation:', sample_sigma)


# Our estimate for the distribution therefore looks like this:



base = np.linspace(-50, 50, 100)
normal = sp.stats.norm.pdf(base, sample_mean, sample_sigma)
lower_bound = sample_mean - (2.58 * sample_sigma)
upper_bound = sample_mean + (2.58 * sample_sigma)
anomalous = np.logical_or(base < [lower_bound]*100, base > [upper_bound]*100)

def plot_est_for_dist():
  plt.plot(base, normal)
  plt.fill_between(base, normal, where=anomalous, color=[1, 0, 0, 0.4])
  plt.xlim(-50, 50)
  plt.show()

plot_est_for_dist()
print('Lower Bound:', lower_bound)
print('Upper Bound:', upper_bound)


# Now we just have to decide on some 'epsilon' value, which dictates our probability threshold for anomalous events.  If we set epsilon to .01, we're saying that any draw for which there's a probability of 1% or less that it given the above distribution should be marked as anomalous.  These values are the upper and lower bounds for what we consider 'normal', and are represented in the graphs above by the area shaded in red.

# Let's look at two sample draws to see if they're anomalous.


def plot_sample_anomalous_draws():
  plt.scatter(X1, np.zeros(N), marker='x', c=plotBlue)
  plt.xlim(-50, 50)
  plt.scatter(-29, 0, marker='x', color='red', s=150, linewidths=3)
  plt.scatter(17, 0, marker='x', color='green', s=150, linewidths=3)
  plt.axvline(lower_bound, ymin=.25, ymax=.75, color='red', linewidth=1)
  plt.axvline(upper_bound, ymin=.25, ymax=.75, color='red', linewidth=1)
  plt.show()

plot_sample_anomalous_draws()

# Here we can see that the red draw exceeds the lower bound, and would therefore come up as anomalous, whereas the green draw falls within the normal range.
# 
# Note that we’re losing some uncertainty by doing it this way.  We’re using the sample mean and standard deviation directly as estimates for the population mean and standard deviation, but of course there is some uncertainty in those estimates.  This model has no mechanism for preserving that uncertainty; we get the same probability estimate for any given event regardless of how certain we are about our estimates for those parameters.

# ### Non-normally distributed single variable
# 
# In this section, we’ll look at a slightly more complicated case, in which we have observations with a single feature that are not normally distributed.  This will demonstrate two additional shortcomings of the standard approach to anomaly detection.  First, we’ll have to manually manipulate the data to ‘look’ normal.  Second, there is no simple way to encode pre-existing knowledge about the distributions into the model.
# 
# Imagine that our observations can only take on positive values, which is a common restriction. For simplicity, let's use the same observations we used earlier, but we'll simply drop all negative observations.



X2 = X1[X1 > 0]

def plot_pos_dist():
  plt.hist(X2, bins=30)
  plt.xlim(-50, 50)
  plt.show()

plot_pos_dist()

# Just looking at the data, it seems that it no longer makes sense to model these observations as normally distributed.  Let's see what happens if we do try to model this as a normal distribution:



sample_mean = X2.mean()
sample_sigma = X2.std()
base = np.linspace(-50, 50, 100)
normal = sp.stats.norm.pdf(base, sample_mean, sample_sigma)
lower_bound = sample_mean - (2.58 * sample_sigma)
upper_bound = sample_mean + (2.58 * sample_sigma)
anomalous = np.logical_or(base < [lower_bound]*100, base > [upper_bound]*100)

def plot_norm_dist_with_pos():
  plt.hist(X2, bins=30, normed=True, zorder=1)
  plt.fill_between(base, normal, where=anomalous, color=[1, 0, 0, 0.4], zorder=2)
  plt.plot(base, normal, color='black', zorder=3)
  plt.xlim(-50, 50)
  plt.show()
  print('Lower Bound:', lower_bound)
  print('Upper Bound:', upper_bound)

plot_norm_dist_with_pos()

# Clearly this distribution does not fit our data!  Most obviously, it wouldn't declare an observation anomalous until it was less than ~ -8.9, while we already know that anything less than 0 is highly anomalous.  When we get into bayesian analysis and probabilistic programming we'll see how we can encode this prior knowledge into our models.

# In the meanwhile, one common work around is to transform the observations until they look vaguely like they come from a normal distribution.  Common approaches for positive distributions include taking the logarithm of every observation, or raising it to a power less than 1.  With this data, it turns out that raising each observation to 0.55 produces something roughly normal.

# Though this is common practice, especially in low-lift anomaly detection systems, eyeballing distribution transformations should make you suspicious.  It results in haphazard analysis and fuzzy statistical reasoning.  Again, probabilistic programming will give us the tools to make sure that all parameters we use in our model are calculated using legitimate statistical methods.



X3 = X2 ** 0.55
def plot_adjusted_hist():
  plt.hist(X3, bins=30)
  plt.show()

plot_adjusted_hist()

# We won't go through it here, but the next steps would be to run the same kind of analysis we ran on the data that actually was normally distributed, and to then check whether a new observation to the power 0.55 seemed anomalous.

# ### Multiple Independently Distributed Normal Variables
# 
# So far we've only been looking at observations with a single feature. We'll now expand our analysis to multiple variables. Initially we will assume that they are independently normal distributed. That is, that each feature is normally distributed, and there is no correlation between them.  Though this is still a simple example, this simple multi-dimensional case will set the stage for evaluating how the traditional methods of anomaly detection perform on more realistic data. 



N = 1000
X1 = np.random.normal(4, 12, N)
X2 = np.random.normal(9, 5, N)
def plot_multi_var_hist():
  plt.scatter(X1, X2, c=plotBlue)
  plt.show()

plot_multi_var_hist()


# As before, we can estimate the means and standard deviations of the normal distributions through the samples.



x1_sample_mean = X1.mean()
x2_sample_mean = X2.mean()
x1_sample_sigma = X1.std()
x2_sample_sigma = X2.std()
def print_stats1():
  print('Sample Mean 1:', x1_sample_mean)
  print('Sample Mean 2:', x2_sample_mean)
  print('Sample Standard Deviation 1:', x1_sample_sigma)
  print('Sample Standard Deviation 2:', x2_sample_sigma)

print_stats1()

# As we would expect, these are not far from the actual values we used to generate the data.
# 
# Next, let's look at a heatmap of where we would expect to find observations given the joint probability distributions implied by these distributions.



delta = 0.025
x1 = np.arange(-40, 50, delta)
x2 = np.arange(-40, 50, delta)

def plot_heatmap_milti_var():
  x, y = np.meshgrid(x1, x2)
  z = plt.mlab.bivariate_normal(x, y, x1_sample_sigma, x2_sample_sigma, x1_sample_mean, x2_sample_mean)
  plt.contourf(x, y, z, cmap='Blues_r')
  thinned_points = np.array([n in np.random.choice(N, 300) for n in range(N)])
  plt.scatter(X1[thinned_points], X2[thinned_points], c='gray')
  plt.show()

plot_heatmap_milti_var()

# Because the two variables are independent, we get this nice concentric circle shape, where as we move in towards the means, we're increasingly likely to draw an observation with those features. As we move away, we're less likely to see an observation with features at those values. We might, for instance, decide that anything in the dark-blue region is anomalous.
# 
# Note that because the distribution of the vertical feature has a smaller variance, the area of high probability is much thinner vertically than it is horizontally.

# ### Multiple Jointly Distributed Features
# 
# In this section, we’ll look at some slightly more realistic data.  We will try to use some of the methods we’ve built up to this point to tackle this more complex data.  
# 
# To build up more of an intuition of what’s going on, let's say that we're observing database transactions, and for each observation we record the latency and the average number of concurrent connections over the course of the transaction.
# 
# We expect these to be positively correlated - the database takes longer to process any given query as the number of active connections grows.  
# 
# We also know that neither of these features can be negative - you cannot have fewer than zero connections to the database, and you cannot finish a transaction before you start it (negative latency).



def positive_support_normal(mean, sigma, n):
    xs = np.random.normal(mean, sigma, n)
    for i, num in enumerate(xs):
        while num < 0:
            num = np.random.normal(mean[i], sigma)
        xs[i] = num
    return xs
    
N = 1000

mu_cons = 10
sigma_cons = 6
sigma_latency = 20
beta = 3

cons = positive_support_normal(np.array([mu_cons]*N), sigma_cons, N)
latency = positive_support_normal(beta * cons, sigma_latency, N)
ax = sns.jointplot('cons', 'latency', pd.DataFrame({'cons': cons, 'latency': latency}))


# Now if we use our previous method - where we treat each feature as independently normally distributed - to think about the probability of finding observations with a particular combination of features, we'll see something like this:



delta = 0.025
x1 = np.arange(-5, 35, delta)
x2 = np.arange(-20, 140, delta)

def plot_prob_obser_comb_features():
  x, y = np.meshgrid(x1, x2)
  z = plt.mlab.bivariate_normal(x, y, cons.std(), latency.std(), cons.mean(), latency.mean())
  plt.contourf(x, y, z, cmap='Blues_r')
  plt.scatter(cons, latency, c='gray')
  plt.ylabel('Latency')
  plt.xlabel('Connections')
  plt.show()

plot_prob_obser_comb_features()

# The problem is that we're missing something about the relationship between the two features.  We _know_ that latency depends on the number of concurrent connections, but it's difficult to account for that in our model.
# 
# There are a few ways to address this issue.  For example, we could engineer a new feature from the existing data that might capture some of the correlation between the two, and treat that as independently normally distributed as well. Alternatively, we could use the observed data to estimate the covariance relation between the two features and keep trying to model them as a multivariate normal distribution.
# 
# Any of these steps quickly get complicated and, more importantly, imprecise.  So far we've only explored relatively simple distributions with very small numbers of features.  Each feature needs to be  be transformed somehow to the normal.  We might need to build new features by hand to improve our models properties, and most often we'll assume that if features are correlated, they at least have joint normal distributions.
# 
# Assumptions about distributions are a good place to start, but we should be able to learn from the data in a more robust and systematic way than eyeballing the value of added features or distribution transformations.
# 
# That's exactly where Bayesian modeling and probabilistic programming come in.

# ### Bayesian modeling 
# 
# Bayesian modeling is a way of calculating the likelihood of some observation, given the data you've already seen.  It allows you to declare your statistical beliefs about what your data _should_ look like before you look at the data.  These beliefs are known as 'priors'.  You can declare one for each of the parameters you're interested in estimating. Bayesian modeling looks at these priors and the observed data, and calculates a distribution for each of those parameters.  These are known as 'posterior distributions'.  
# 
# Part of the benefit of probabilistic programming, relative to comparable Bayesian modeling in the past, is that you don't need to know anything about how that calculation happens.
# 
# ### Probabilistic Programming
# 
# Roughly speaking, the steps involved in probabilistic programming are as follows:
# 
# 1. Declare your prior beliefs about what the data looks like.
# 2. Feed in your data.
# 3. A sampling algorithm will return a posterior distribution for each of the parameters in your model.
# 


# Probabilistic programming languages enable you to state your priors beliefs and your model with elegant, statistical syntax.  
# 
# The more data you have to learn from, the less your prior beliefs matter, and generally the tighter the distributions you'll get for your model parameters.
# 
# There are several advantages here.  First, you get a formal way to encode your knowledge about the data into your model.  As we saw earlier, this is especially useful when you know that there is no chance at all of certain observations.
# 
# Second, there are no limitations on which distributions we can use to model our world, or on the hierarchical relations we claim they have.  Which  means, among other things, that and we don't have to estimate data manipulations to make things look right.
# 
# Third, we get to propagate our uncertainty.  For example, instead of simply estimating the mean of the population distribution as the sample mean, we estimate the population mean as a distribution itself, and the nature of this uncertainty will affect our final model of the data.
# 
# It's simpler than ever to build a complex model of the world.

