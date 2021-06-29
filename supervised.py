import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt



FILE_NAMES = ["./rt-polarity.pos", "./rt-polarity.neg"]
labeled_data_sets = []

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    
TOTAL_LINE = 5332*2
TRAIN_SIZE = int(0.7*TOTAL_LINE)
VAL_SIZE = int(0.2*TOTAL_LINE)
TEST_SIZE = int(0.1*TOTAL_LINE)
BATCH_SIZE = 512

all_labeled_data = all_labeled_data.shuffle(20000)
train_ds = all_labeled_data.take(TRAIN_SIZE).batch(BATCH_SIZE)
test_ds = all_labeled_data.skip(TRAIN_SIZE)
val_ds = test_ds.skip(TEST_SIZE).batch(BATCH_SIZE)
test_ds = test_ds.take(TEST_SIZE).batch(TEST_SIZE)

print("the size of validation set:", VAL_SIZE,"\nthe size of test set:", TEST_SIZE)


class Classifier_Model(tf.keras.Model):
    def __init__(self, b_filepath="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                 pre_filepath="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                 num_classes=2):

        super(Classifier_Model, self).__init__()
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
        if training:
            dropout = tf.keras.layers.Dropout(0.1)
            net = dropout(net)
        return self.nn(net)
        
model = Classifier_Model()  
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5))

history = model.fit(train_ds, validation_data=val_ds, epochs=40, verbose=1)
