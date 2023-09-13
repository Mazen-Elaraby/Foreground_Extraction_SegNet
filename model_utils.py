import tensorflow as tf
import keras.backend as K

#custom loss function to filter non-ROI
def c_loss(y_true, y_pred):

    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)

    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

#custom accuracy metric to filter non-ROI
def c_acc(y_true, y_pred):
    
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)

    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)