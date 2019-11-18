import numpy as np
import pandas as pd
import edward as ed
from edward.models import Bernoulli, Normal, MultivariateNormalTriL
from edward.util import rbf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load and preprocess
data = pd.read_csv("genus_data.csv")
varbvs_prior = np.genfromtxt("genus_varbvs_prior.csv")
str_col = [isinstance(data.iloc[0, i], str) for i in range(data.shape[1])]
X = data.loc[:, [not i for i in str_col]]
row_idx = [i for i in range(X.shape[0])]
np.random.shuffle(row_idx)
X = X.iloc[row_idx, :]
X_ndar = np.array(X)
X_arr = X_ndar[:, :-1]
X_arr = StandardScaler().fit_transform(X_arr)
y_arr = X_ndar[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=.5, random_state=1)

# doing this because I don't know how to make the testing work otherwise
# it seems like the test and training data need to have the same N
X_test, y_test = X_test[:-1, :], y_test[:-1]

# unfortunately not sure how to make the linear kernel work at this moment
N, P = X_train.shape
X_tf = tf.placeholder(tf.float32, [N, P])

# latent stochastic function
# ok so here in the loc position is where we can get (x *element-wise* b)
b = Bernoulli(varbvs_prior, dtype=np.float32) # prior from varbvs
gp_mu = tf.reduce_mean(tf.multiply(X_tf, tf.reshape(tf.tile(b, [N]), [N, P])), 1) # mean for prior over GP

f = MultivariateNormalTriL(
    loc=gp_mu,
    scale_tril=tf.cholesky(rbf(X_tf)) # uses rbf kernel for covariance of GP for now
)

qf = Normal(loc=tf.get_variable("qf/loc", [N]), scale=tf.nn.softplus(tf.get_variable("qf/scale", [N])))

# respose
y_tf = Bernoulli(logits=f)

# inference
infer = ed.KLqp({f: qf}, data={X_tf: X_train, y_tf: y_train})
infer.run(n_samples=3, n_iter=5000)

# criticism
y_post = ed.copy(y_tf, {f: qf})
ed.evaluate('binary_accuracy', data={X_tf: X_test, y_post: y_test})
