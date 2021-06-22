import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


def labeler(example, index):
    return example, tf.cast(index, tf.int64)

LABEL_FILES = ["./label_pos.txt", "./label_neg.txt"] 
labeled_data_sets = []

for i, file_name in enumerate(LABEL_FILES):
    lines_dataset = tf.data.TextLineDataset(str(file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    
 
UNLABEL_FILES = ["./unlabel_pos.txt", "./unlabel_neg.txt"]

unlabeled_data_sets = []
for file_name in UNLABEL_FILES:
    lines_dataset = tf.data.TextLineDataset(str(file_name))
    unlabeled_data_sets.append(lines_dataset)

all_unlabeled_data =  unlabeled_data_sets[0]
for unlabeled_dataset in unlabeled_data_sets[1:]:
    all_unlabeled_data = all_unlabeled_data.concatenate(unlabeled_dataset)
    
TRAIN_SIZE = 1024
VAL_SIZE = 256
TEST_SIZE = 256
BATCH_SIZE=64

all_labeled_data = all_labeled_data.shuffle(16384, seed=3114)
train_ds = all_labeled_data.take(TRAIN_SIZE).batch(BATCH_SIZE)
test_ds = all_labeled_data.skip(TRAIN_SIZE)
val_ds = test_ds.skip(TEST_SIZE).batch(VAL_SIZE)
test_ds = test_ds.take(TEST_SIZE).batch(TEST_SIZE)

all_train_ds = tf.data.Dataset.zip((train_ds.shuffle(8192), 
                                    all_unlabeled_data.shuffle(8192).batch(572)))


class Network(tf.keras.Model):
    def __init__(self, b_filepath="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                 pre_filepath="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                 num_classes=2):

        super(Network, self).__init__()
        self.preprocessor = hub.KerasLayer(pre_filepath, name='preprocessing')
        self.bert_model = hub.KerasLayer(b_filepath, 
                                         trainable=False, name='BERT_encoder')
        self.reshape_layer = tf.keras.layers.Dense(768, activation='tanh', name='reshape')
        self.nn = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')

    @tf.function
    def call(self, inputs, training, **kwargs):
        if len(inputs.shape) == 2: # (batch, 1)
            x = self.preprocessor(tf.squeeze(inputs, axis=-1))
        else:
            x = self.preprocessor(inputs)
        outputs = self.bert_model(x)
        net = tf.concat([outputs['encoder_outputs'][i][:, 0, :] for i in range(-4, 0 ,1)], axis=-1)
        net = self.reshape_layer(net)    
        return self.nn(net)

teacher = Network()
student = Network()

acc_metric = tf.metrics.SparseCategoricalAccuracy(name='accuracy')


class TSNetwork(tf.keras.Model):
    def __init__(self, teacher, student):
        super(TSNetwork, self).__init__()
        self.teacher = teacher
        self.student = student
    
    def compile(self, t_optimizer, s_optimizer, loss_fn, **kwagrs):
        super(TSNetwork, self).compile()
        self.t_optimizer = t_optimizer
        self.s_optimizer = s_optimizer
        self.loss_fn = loss_fn

    @tf.function    
    def train_step(self, data):
        labeled_data, X_unlabel = data
        X_label, y = labeled_data
        
        # Train the student using the pseudo label
        # pseudo label viewed as real target
        pseudo = tf.math.argmax(self.teacher(X_unlabel), axis=-1)
        with tf.GradientTape() as tape:
            prediction = self.student(X_unlabel)
            s_unsupervised_loss = self.loss_fn(pseudo, prediction)
        grad_s_unsupervised = tape.gradient(s_unsupervised_loss, self.student.trainable_weights)
        self.s_optimizer.apply_gradients(zip(grad_s_unsupervised, self.student.trainable_weights)) 

        # Train the teacher
        # Compute teacher's feedback coefficent
        with tf.GradientTape() as tape:
            prediction_label = self.student(X_label)
            s_supervised_loss = self.loss_fn(y, prediction_label)
        grad_s_supervised = tape.gradient(s_supervised_loss, self.student.trainable_weights)

        h = []
        for i in range(len(grad_s_supervised)):
            if len(grad_s_supervised[i].shape) == 2:
                gradient = self.s_optimizer.learning_rate * tf.matmul(
                                                            grad_s_supervised[i], 
                                                            grad_s_unsupervised[i], 
                                                            transpose_b=True)
            else:
                gradient = self.s_optimizer.learning_rate * tf.math.reduce_sum(grad_s_supervised[i] * grad_s_unsupervised[i])
            h.append(gradient)

        # Compute teacher's gradient from student's feedback
        with tf.GradientTape() as tape:
            prediction = self.teacher(X_unlabel)
            t_unsupervised_loss = self.loss_fn(pseudo, prediction)
        
        grad_t_unsupervised = tape.gradient(t_unsupervised_loss, self.teacher.trainable_weights)
        grad_t = []
        for i in range(len(h)): # len(h) == 4
            if len(h[i].shape)==2:
                value = tf.matmul(h[i], grad_t_unsupervised[i]) 
            else:
                value = h[i] * grad_t_unsupervised[i]
            grad_t.append(value)

        # Compute the teacher's gradient on labeled data
        with tf.GradientTape() as tape:
            prediction = self.teacher(X_label)
            t_supervised_loss = self.loss_fn(y, prediction)
        grad_t_supervised = tape.gradient(t_supervised_loss, self.teacher.trainable_weights)
        total_grad_t = [x + y for x, y in zip(grad_t, grad_t_supervised)]
        self.t_optimizer.apply_gradients(zip(total_grad_t, self.teacher.trainable_weights))

        # Compute metrics
        acc_metric.update_state(y, prediction_label)

        return {"accuracy": acc_metric.result(),
                "s_unsupervised_loss": s_unsupervised_loss, "s_supervised_loss": s_supervised_loss, 
                "t_unsupervised_loss": t_unsupervised_loss, "t_supervised_loss": t_supervised_loss}

    @tf.function
    def test_step(self, data):
        X, y = data 
        y_pred = self.student(X, training=False)
        acc_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [acc_metric]

tsnet = TSNetwork(teacher, student)
tsnet.compile(
    t_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    s_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
history = tsnet.fit(all_train_ds, validation_data=val_ds, epochs=40, verbose=1)

