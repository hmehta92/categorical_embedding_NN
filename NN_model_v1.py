################## keras categorical and numerical features ##############
def get_input_features(df):
    X['numerical'] = np.array(df[numerical_features])
    for cat in categorical_features:
        X[cat] = np.array(df[cat])
    return X
	
	
# Build model
def sparse_top_5_cat_acc(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=N)

def build_nn_model():
    clear_session()
    ############################# categorical ########################
    categorical_inputs = []
    for cat in categorical_features:
        categorical_inputs.append(Input(shape=[1], name=cat))
    ################################################# Model 1 #################################################
    categorical_embeddings = []
    for i, cat in enumerate(categorical_features):
        result = Embedding(category_counts[cat], int(np.log1p(category_counts[cat]) + 1), \
                                                name = cat + "_embd")(categorical_inputs[i])
        categorical_embeddings.append(result)
    ################################# concatenate all categorical layers ####################################
    categorical_logits = Concatenate(name = 'categorical_conc')([Flatten()(SpatialDropout1D(spatial_drop_rate)(cat_emb)) for cat_emb in categorical_embeddings])
    model_2 = Model(categorical_inputs,categorical_logits)
    
    ################################################# Model 2, numerical telemetry ########################################
    numerical_inputs = Input(shape = [len(numerical_features)], name = 'numerical')
    x = Dense(64, activation = 'relu')(numerical_inputs)
    x = BatchNormalization()(x)
    model_3 = Model(numerical_inputs, x)
#   x = Dropout(.05)
    concatenated = concatenate([categorical_logits, x])
    out =  Dense(number_of_classes+1, activation='softmax', name = 'output')(concatenated)
    model = Model(inputs = [numerical_inputs] + categorical_inputs, outputs = out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])
    model.summary()
    return model
