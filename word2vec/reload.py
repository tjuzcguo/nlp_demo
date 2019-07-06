import tensorflow as tf

tf.reset_default_graph()

v1 = tf.get_variable("emb", shape=[4035, 200])

saver = tf.train.import_meta_graph("/Users/guozongchao/Desktop/Material/machine/model.ckpt-592298.meta")

with tf.Session() as sess:
    saver.restore(sess, "/Users/guozongchao/Desktop/Material/machine/model.ckpt")  # 注意路径写法
    print(sess.run(tf.get_default_graph().get_tensor_by_name("emb")))  # [ 3.]



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   #saver.restore(sess, "/Users/guozongchao/Desktop/Material/machine/model.ckpt")
#   model_file = tf.train.latest_checkpoint('/Users/guozongchao/Desktop/Material/machine/')
#   saver.restore(sess, model_file)
#
#   print("Model restored.")
#   print("v1 : %s" % v1.eval())

# Check the values of the variables
