# quora

This was built for a Quora question pairing contest on Kaggle.

This was about matching pair of questions on Quora.
Training set had 2 columns - question1 & question2 containing a set of sentences. A target column contained the matching
quotient for each question pair - 1 means same; 0 mean different

These were the steps in the solution:
1. Use Word2vec pre-trained model from Google to embed the words into 300 dimensional tensors
2. For question1 set, average number of words per sentence = 10; so, timesteps = 10
3. For question2 set, average number of words per sentence = 52; so, timesteps ~ 50
4. Build a RNN/LSTM encoder that would reduce/encode the dimensions from 300 to 32 for question1. Number of time sequences = 10
5. Build a RNN/LSTM encoder that would reduce/encode the dimensions from 300 to 32 for question2. Number of time sequences = 50
6. Save the tensors from question1 & question2 encoders
7. Create a tensor whose first column is question1  + question2 tensor, and second column is question1 * question2 tensor
8. Feed this tensor into a MLP (Multi layer perceptron) whose target is the matching quotient of question1 & question2 in
training set
