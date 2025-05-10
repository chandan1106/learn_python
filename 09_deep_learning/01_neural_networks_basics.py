"""
Deep Learning Fundamentals: Neural Networks Basics
"""

# This file provides an introduction to neural networks and deep learning
# In a real environment, you would need to install TensorFlow: pip install tensorflow
print("Note: This code assumes TensorFlow is installed. If you get an ImportError, install it with: pip install tensorflow")

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    print("TensorFlow successfully imported! Version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed. Please install it with: pip install tensorflow")
    # Continue with the theoretical part
    pass

# ===== INTRODUCTION TO NEURAL NETWORKS =====
print("\n===== INTRODUCTION TO NEURAL NETWORKS =====")
"""
Neural Networks are computing systems inspired by the biological neural networks in animal brains.
They consist of artificial neurons that process and transmit information.

Key Components:
1. Neurons: Basic computational units that receive inputs, apply an activation function, and produce an output
2. Layers: Groups of neurons that process information
   - Input Layer: Receives the initial data
   - Hidden Layers: Process the information
   - Output Layer: Produces the final result
3. Weights: Parameters that determine the strength of connections between neurons
4. Biases: Additional parameters that allow shifting the activation function
5. Activation Functions: Non-linear functions that determine the output of a neuron
6. Loss Function: Measures how well the network is performing
7. Optimizer: Algorithm that adjusts weights to minimize the loss function

Types of Neural Networks:
1. Feedforward Neural Networks (FNN): Information flows in one direction
2. Convolutional Neural Networks (CNN): Specialized for processing grid-like data (e.g., images)
3. Recurrent Neural Networks (RNN): Have connections that form cycles, useful for sequential data
4. Long Short-Term Memory (LSTM): Special RNN that can learn long-term dependencies
5. Generative Adversarial Networks (GAN): Two networks compete to generate realistic data
6. Transformers: Based on self-attention mechanisms, powerful for NLP tasks
"""

# ===== ACTIVATION FUNCTIONS =====
print("\n===== ACTIVATION FUNCTIONS =====")
"""
Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns.

Common Activation Functions:

1. Sigmoid: σ(x) = 1 / (1 + e^(-x))
   - Output range: (0, 1)
   - Used for binary classification output layers
   - Pros: Smooth gradient, output as probability
   - Cons: Vanishing gradient problem, not zero-centered

2. Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Output range: (-1, 1)
   - Zero-centered
   - Pros: Zero-centered, stronger gradients than sigmoid
   - Cons: Still has vanishing gradient problem

3. ReLU (Rectified Linear Unit): f(x) = max(0, x)
   - Output range: [0, ∞)
   - Pros: Computationally efficient, reduces vanishing gradient
   - Cons: "Dying ReLU" problem (neurons can become inactive)

4. Leaky ReLU: f(x) = max(αx, x) where α is a small constant
   - Output range: (-∞, ∞)
   - Pros: Prevents dying ReLU problem
   - Cons: Results can be inconsistent

5. Softmax: σ(z)_i = e^(z_i) / Σ(e^(z_j))
   - Converts a vector of values to a probability distribution
   - Used for multi-class classification output layers
   - Output range: (0, 1) for each element, sum = 1
"""

# Visualize activation functions if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid)
    plt.title('Sigmoid')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh)
    plt.title('Tanh')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, relu)
    plt.title('ReLU')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu)
    plt.title('Leaky ReLU')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    print("Activation functions visualization saved as 'activation_functions.png'")
except ImportError:
    print("Matplotlib is not installed. Skipping visualization.")

# ===== BUILDING A SIMPLE NEURAL NETWORK =====
print("\n===== BUILDING A SIMPLE NEURAL NETWORK =====")

# Check if TensorFlow is available
if 'tf' in globals():
    print("Building a simple neural network with TensorFlow/Keras...")
    
    # Load a simple dataset (MNIST)
    print("Loading the MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Build a simple feedforward neural network
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Train the model (with a small subset for demonstration)
    print("\nTraining the model (with a small subset)...")
    history = model.fit(
        x_train[:5000], y_train[:5000],
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(x_test[:5])
    print("Predictions for the first 5 test images:")
    for i, pred in enumerate(predictions):
        print(f"Image {i+1}: Predicted class: {np.argmax(pred)}, Actual class: {y_test[i]}")
else:
    print("TensorFlow is not available. Skipping the practical neural network example.")
    print("Here's the code you would use to build a simple neural network with TensorFlow/Keras:")
    print("""
    # Load a simple dataset (MNIST)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Build a simple feedforward neural network
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    """)

# ===== NEURAL NETWORK ARCHITECTURE =====
print("\n===== NEURAL NETWORK ARCHITECTURE =====")
"""
Neural Network Architecture refers to the structure and organization of neurons in a network.

Key Architectural Decisions:
1. Number of Layers:
   - Shallow Networks: Few layers
   - Deep Networks: Many layers (deep learning)

2. Layer Types:
   - Dense/Fully Connected: Each neuron connects to all neurons in the next layer
   - Convolutional: Specialized for grid-like data (e.g., images)
   - Recurrent: For sequential data with temporal dependencies
   - Embedding: Converts discrete inputs to continuous vectors
   - Normalization: Batch normalization, layer normalization
   - Pooling: Reduces dimensionality (e.g., max pooling)
   - Dropout: Randomly deactivates neurons during training to prevent overfitting

3. Number of Neurons per Layer:
   - Input layer: Matches the number of features
   - Hidden layers: Depends on problem complexity
   - Output layer: Matches the number of classes (classification) or outputs (regression)

4. Connectivity Patterns:
   - Feedforward: Information flows in one direction
   - Skip connections: Connect non-adjacent layers (e.g., ResNet)
   - Recurrent connections: Form cycles in the network

5. Activation Functions:
   - Hidden layers: ReLU, Leaky ReLU, tanh
   - Output layer: Sigmoid (binary), Softmax (multi-class), Linear (regression)
"""

# ===== TRAINING NEURAL NETWORKS =====
print("\n===== TRAINING NEURAL NETWORKS =====")
"""
Training a neural network involves adjusting its weights to minimize a loss function.

Key Training Concepts:

1. Loss Functions:
   - Mean Squared Error (MSE): For regression problems
   - Binary Cross-Entropy: For binary classification
   - Categorical Cross-Entropy: For multi-class classification
   - Sparse Categorical Cross-Entropy: Same as above, but with integer labels

2. Backpropagation:
   - Algorithm to compute gradients of the loss with respect to weights
   - Uses the chain rule of calculus
   - Propagates error backward through the network

3. Optimizers:
   - Stochastic Gradient Descent (SGD): Simple, but can be slow
   - SGD with Momentum: Adds momentum to avoid local minima
   - Adam: Adaptive learning rates, combines momentum and RMSProp
   - RMSProp: Adapts learning rates based on recent gradients
   - Adagrad: Adapts learning rates based on parameter history

4. Learning Rate:
   - Controls how much weights are adjusted in each step
   - Too high: May overshoot or diverge
   - Too low: Slow convergence or stuck in local minima
   - Learning rate schedules: Decrease learning rate over time

5. Batch Size:
   - Number of samples processed before updating weights
   - Larger batches: More stable gradients, but require more memory
   - Smaller batches: More noise, but can help escape local minima

6. Epochs:
   - Number of complete passes through the training dataset
   - Too few: Underfitting
   - Too many: Overfitting

7. Regularization:
   - L1/L2 regularization: Add penalty for large weights
   - Dropout: Randomly deactivate neurons during training
   - Early stopping: Stop training when validation performance stops improving
   - Data augmentation: Create new training samples by modifying existing ones
"""

# ===== CONVOLUTIONAL NEURAL NETWORKS =====
print("\n===== CONVOLUTIONAL NEURAL NETWORKS =====")
"""
Convolutional Neural Networks (CNNs) are specialized for processing grid-like data, such as images.

Key Components:

1. Convolutional Layers:
   - Apply filters to detect features (edges, textures, patterns)
   - Parameters: number of filters, filter size, stride, padding
   - Each filter produces a feature map
   - Shared weights reduce parameters and capture spatial patterns

2. Pooling Layers:
   - Reduce spatial dimensions (downsampling)
   - Common types: Max pooling, average pooling
   - Helps achieve spatial invariance

3. CNN Architecture:
   - Input layer: Raw image (height × width × channels)
   - Alternating convolutional and pooling layers
   - Flatten layer: Convert 2D feature maps to 1D vector
   - Dense layers: Final classification or regression

4. Popular CNN Architectures:
   - LeNet-5: Early CNN for digit recognition
   - AlexNet: Breakthrough in image classification (2012)
   - VGG: Simple architecture with small filters
   - ResNet: Introduced skip connections to train very deep networks
   - Inception/GoogLeNet: Uses inception modules with parallel convolutions
   - MobileNet: Lightweight for mobile and embedded devices
   - EfficientNet: Balanced scaling of network dimensions
"""

# Example of building a CNN with TensorFlow/Keras
if 'tf' in globals():
    print("\nBuilding a simple CNN with TensorFlow/Keras...")
    
    # Load the MNIST dataset (as images this time)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    # Build a simple CNN
    cnn_model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    cnn_model.summary()
    
    # Train the model (with a small subset for demonstration)
    print("\nTraining the CNN (with a small subset)...")
    cnn_history = cnn_model.fit(
        x_train[:5000], y_train[:5000],
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the CNN...")
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test, verbose=0)
    print(f"CNN Test accuracy: {cnn_test_acc:.4f}")
else:
    print("\nTensorFlow is not available. Skipping the CNN example.")
    print("Here's the code you would use to build a simple CNN with TensorFlow/Keras:")
    print("""
    # Load the MNIST dataset (as images)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    # Build a simple CNN
    cnn_model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    cnn_history = cnn_model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )
    
    # Evaluate the model
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test)
    print(f"CNN Test accuracy: {cnn_test_acc:.4f}")
    """)

# ===== RECURRENT NEURAL NETWORKS =====
print("\n===== RECURRENT NEURAL NETWORKS =====")
"""
Recurrent Neural Networks (RNNs) are designed for sequential data by maintaining an internal state.

Key Concepts:

1. Basic RNN:
   - Has connections that form cycles
   - Maintains a hidden state that captures information from previous steps
   - Same weights are used at each time step (parameter sharing)
   - Suffers from vanishing/exploding gradient problems

2. Long Short-Term Memory (LSTM):
   - Special RNN architecture with gating mechanisms
   - Gates: input, forget, output
   - Cell state: Long-term memory
   - Hidden state: Short-term memory
   - Better at capturing long-term dependencies

3. Gated Recurrent Unit (GRU):
   - Simplified version of LSTM
   - Has reset and update gates
   - Fewer parameters than LSTM
   - Often similar performance to LSTM

4. Bidirectional RNNs:
   - Process sequences in both forward and backward directions
   - Capture context from both past and future
   - Useful for tasks where entire sequence is available

5. Applications:
   - Natural language processing
   - Speech recognition
   - Time series prediction
   - Music generation
   - Video analysis
"""

# Example of building an RNN with TensorFlow/Keras
if 'tf' in globals():
    print("\nBuilding a simple RNN with TensorFlow/Keras...")
    
    # Create a simple sequence dataset
    # Generate sequences of 20 integers, predict the next integer
    def generate_sequences(n_sequences=1000, seq_length=20):
        X = np.zeros((n_sequences, seq_length))
        y = np.zeros((n_sequences, 1))
        
        for i in range(n_sequences):
            # Generate a random starting point
            start = np.random.randint(1, 50)
            # Generate an arithmetic sequence
            seq = np.arange(start, start + seq_length + 1)
            X[i] = seq[:seq_length]
            y[i] = seq[seq_length]
        
        # Normalize
        X = X / 100.0
        y = y / 100.0
        
        return X, y
    
    # Generate data
    X_train, y_train = generate_sequences(1000)
    X_test, y_test = generate_sequences(200)
    
    # Reshape for RNN [samples, time steps, features]
    X_train = X_train.reshape(-1, 20, 1)
    X_test = X_test.reshape(-1, 20, 1)
    
    # Build a simple RNN model
    rnn_model = keras.Sequential([
        keras.layers.SimpleRNN(50, activation='relu', input_shape=(20, 1), return_sequences=False),
        keras.layers.Dense(1)
    ])
    
    # Compile the model
    rnn_model.compile(optimizer='adam', loss='mse')
    
    # Display model summary
    rnn_model.summary()
    
    # Train the model
    print("\nTraining the RNN...")
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the RNN...")
    rnn_test_loss = rnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"RNN Test loss (MSE): {rnn_test_loss:.6f}")
    
    # Make predictions
    print("\nMaking predictions with the RNN...")
    predictions = rnn_model.predict(X_test[:5])
    for i in range(5):
        actual_seq = X_test[i].flatten() * 100
        predicted = predictions[i][0] * 100
        actual = y_test[i][0] * 100
        print(f"Sequence: {actual_seq}")
        print(f"Actual next number: {actual:.1f}, Predicted: {predicted:.1f}")
        print()
    
    # Build an LSTM model
    print("\nBuilding an LSTM model...")
    lstm_model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', input_shape=(20, 1), return_sequences=False),
        keras.layers.Dense(1)
    ])
    
    # Compile the model
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Display model summary
    lstm_model.summary()
    
    # Train the model
    print("\nTraining the LSTM...")
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the LSTM...")
    lstm_test_loss = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Test loss (MSE): {lstm_test_loss:.6f}")
else:
    print("\nTensorFlow is not available. Skipping the RNN example.")
    print("Here's the code you would use to build a simple RNN with TensorFlow/Keras:")
    print("""
    # Create a simple sequence dataset
    def generate_sequences(n_sequences=1000, seq_length=20):
        X = np.zeros((n_sequences, seq_length))
        y = np.zeros((n_sequences, 1))
        
        for i in range(n_sequences):
            # Generate a random starting point
            start = np.random.randint(1, 50)
            # Generate an arithmetic sequence
            seq = np.arange(start, start + seq_length + 1)
            X[i] = seq[:seq_length]
            y[i] = seq[seq_length]
        
        # Normalize
        X = X / 100.0
        y = y / 100.0
        
        return X, y
    
    # Generate data
    X_train, y_train = generate_sequences(1000)
    X_test, y_test = generate_sequences(200)
    
    # Reshape for RNN [samples, time steps, features]
    X_train = X_train.reshape(-1, 20, 1)
    X_test = X_test.reshape(-1, 20, 1)
    
    # Build a simple RNN model
    rnn_model = keras.Sequential([
        keras.layers.SimpleRNN(50, activation='relu', input_shape=(20, 1), return_sequences=False),
        keras.layers.Dense(1)
    ])
    
    # Compile the model
    rnn_model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    # Build an LSTM model
    lstm_model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', input_shape=(20, 1), return_sequences=False),
        keras.layers.Dense(1)
    ])
    
    # Compile the model
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    """)

# ===== DEEP LEARNING BEST PRACTICES =====
print("\n===== DEEP LEARNING BEST PRACTICES =====")
"""
Best Practices for Deep Learning:

1. Data Preparation:
   - Normalize/standardize inputs
   - Handle missing values
   - Split data into training, validation, and test sets
   - Use data augmentation for limited data

2. Architecture Design:
   - Start with proven architectures
   - Use appropriate layer types for your data
   - Consider the depth and width of your network
   - Use skip connections for very deep networks

3. Training:
   - Use mini-batch gradient descent
   - Monitor training and validation metrics
   - Implement early stopping
   - Use learning rate schedules or adaptive optimizers
   - Save checkpoints of your model

4. Regularization:
   - Apply dropout
   - Use L1/L2 regularization
   - Implement batch normalization
   - Consider data augmentation

5. Hyperparameter Tuning:
   - Learning rate
   - Batch size
   - Number of layers and neurons
   - Regularization strength
   - Optimizer parameters

6. Debugging:
   - Start with a simple model that works
   - Add complexity incrementally
   - Visualize activations and gradients
   - Use TensorBoard or similar tools

7. Deployment:
   - Quantize models for efficiency
   - Consider model pruning
   - Optimize for inference
   - Monitor performance in production
"""

# ===== ADVANCED DEEP LEARNING TOPICS =====
print("\n===== ADVANCED DEEP LEARNING TOPICS =====")
"""
Advanced Topics in Deep Learning:

1. Transfer Learning:
   - Use pre-trained models as starting points
   - Fine-tune on your specific task
   - Especially useful with limited data

2. Generative Models:
   - Generative Adversarial Networks (GANs)
   - Variational Autoencoders (VAEs)
   - Diffusion Models
   - Generate realistic images, text, audio, etc.

3. Reinforcement Learning:
   - Learn through trial and error with rewards
   - Deep Q-Networks (DQN)
   - Policy Gradient methods
   - Applications: games, robotics, recommendation systems

4. Self-Supervised Learning:
   - Learn from unlabeled data by creating supervised tasks
   - Contrastive learning
   - Masked language modeling
   - Reduces dependence on labeled data

5. Attention Mechanisms:
   - Focus on relevant parts of the input
   - Self-attention in Transformers
   - Revolutionized NLP and expanding to other domains

6. Transformers:
   - Based on self-attention
   - No recurrence or convolution
   - Models like BERT, GPT, T5
   - State-of-the-art in NLP and expanding to vision

7. Neural Architecture Search (NAS):
   - Automatically discover optimal architectures
   - Reduces human bias in design
   - Can be computationally expensive

8. Explainable AI (XAI):
   - Interpret and explain model decisions
   - Techniques: SHAP, LIME, Grad-CAM
   - Important for trust and regulatory compliance

9. Federated Learning:
   - Train models across multiple devices while keeping data local
   - Privacy-preserving
   - Challenges: communication efficiency, heterogeneous data

10. Quantum Neural Networks:
    - Leverage quantum computing for neural networks
    - Potential for exponential speedup
    - Still in early research stages
"""

print("\n===== END OF NEURAL NETWORKS BASICS =====")