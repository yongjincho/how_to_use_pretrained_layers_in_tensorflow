# How to use pretrained layers in Tensorflow

This respository explains how to use pretrained layers for the new model in Tensorflow.

The key element which enables this functionality is ```tf.estimator.WarmStartSettings``` class.  You can specify where the pretrained weights and biases of the new model are by using this class.  The details are described in this example codes.

You can see the official document of ```WarmStartSettings``` class at <https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings>.

## 0. Requirements

* Tensorflow >= 1.9
* Python >= 3.5

## 1. Pretrain the base model

This is a simple mnist model, but you should carefully design the layer name to be easily imported by the new model.  In this example, "mnist/hidden/kernel" and "mnist/hidden/bias" will be used by the new model.  As you can see, checkpoint files will be saved in the ```./model/``` directory.

```
$ python mnist.py
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': './model', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x111626e48>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-07-11 23:11:43.536136: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./model/model.ckpt.
INFO:tensorflow:loss = 2.4827523, step = 1
INFO:tensorflow:Saving checkpoints for 100 into ./model/model.ckpt.
INFO:tensorflow:Loss for final step: 1.8764114.
$
```

## 2. Train a new model by using pretrained weights

```mnist_ext.py``` uses the pretrained hidden layer and stacks another hidden layer on it.  The weight and bias of the first layer is specified by following.

```
    warmup = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.model_dir,
            vars_to_warm_start=[
                "mnist/hidden/kernel",
                "mnist/hidden/bias",
            ])

    estimator = tf.estimator.Estimator(
            model_fn,
            FLAGS.model_ext_dir,
            warm_start_from=warmup)
```

```ckpt_to_initialize_from``` is set to the pretrained model directory.  You can also use a specific checkpoint by passing its path.

```vars_to_warm_start``` parameter can be a regular expression or a list of specific variable names.  In Tensorflow 1.8, list parameter is not supported and, therefore, you cannot freeze the imported variables.  In Tensorflow 1.9, however, you can freeze them by passing the list of specific variable names.

In this example, ```mnist/hidden/kernel``` and ```mnist/hidden/bias``` are loaded and are frozen.  As you can see in the following, the variables are initialized from the pretrained model.

```
$ python mnist_ext.py
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': './model_ext', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x118d90278>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Warm-starting with WarmStartSettings: WarmStartSettings(ckpt_to_initialize_from='./model', vars_to_warm_start=['mnist/hidden/kernel', 'mnist/hidden/bias'], var_name_to_vocab_info={}, var_name_to_prev_var_name={})
INFO:tensorflow:Warm-starting from: ('./model',)
INFO:tensorflow:Warm-starting variable: mnist/hidden/kernel; prev_var_name: Unchanged
INFO:tensorflow:Initialize variable mnist/hidden/kernel:0 from checkpoint ./model with mnist/hidden/kernel
INFO:tensorflow:Warm-starting variable: mnist/hidden/bias; prev_var_name: Unchanged
INFO:tensorflow:Initialize variable mnist/hidden/bias:0 from checkpoint ./model with mnist/hidden/bias
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-07-11 23:12:20.801789: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./model_ext/model.ckpt.
INFO:tensorflow:loss = 2.5067844, step = 1
INFO:tensorflow:Saving checkpoints for 100 into ./model_ext/model.ckpt.
INFO:tensorflow:Loss for final step: 2.5439825.
$
```

### 3. Check the importeded variables

You can see the ```mnist/hidden/bias``` variables of the two models are the same.

```
tensorflow_transfer_learning_example$ bash diff.sh
tensor_name:  mnist/hidden/bias
[-2.6944827e-03  2.4477108e-03  4.2932539e-04 -2.7114481e-03
  8.3686592e-04 -5.1927392e-04 -8.3964766e-04  1.1230807e-04
 -8.8314572e-04  4.3607219e-03  1.2164903e-03  3.0878973e-03
 -9.9602353e-04  1.9190852e-03  5.7564303e-04  5.1802403e-05
  9.4812026e-04  1.8225886e-03  6.3254461e-03 -1.1712501e-03]
tensor_name:  mnist/hidden/bias
[-2.6944827e-03  2.4477108e-03  4.2932539e-04 -2.7114481e-03
  8.3686592e-04 -5.1927392e-04 -8.3964766e-04  1.1230807e-04
 -8.8314572e-04  4.3607219e-03  1.2164903e-03  3.0878973e-03
 -9.9602353e-04  1.9190852e-03  5.7564303e-04  5.1802403e-05
  9.4812026e-04  1.8225886e-03  6.3254461e-03 -1.1712501e-03]
```
