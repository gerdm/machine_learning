function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    visible_data = sample_bernoulli(visible_data);

    % Performing Gibbs sampling
    sample_hidden1_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    sample_hidden1 = sample_bernoulli(sample_hidden1_prob);

    sample_visible_prob = hidden_state_to_visible_probabilities(rbm_w, sample_hidden1);
    sample_visible = sample_bernoulli(sample_visible_prob);

    sample_hidden2_prob = visible_state_to_hidden_probabilities(rbm_w, sample_visible);
    sample_hidden2 = sample_bernoulli(sample_hidden2_prob);

    positive_phase = configuration_goodness_gradient(visible_data, sample_hidden1);
    % To reduce the variance, yet for me to show, we compute the gradient consideraing
    % the probability of the sampled visible space to the 'new' hidden state
    negative_phase = configuration_goodness_gradient(sample_visible, sample_hidden2_prob);

    ret = positive_phase - negative_phase;
end
