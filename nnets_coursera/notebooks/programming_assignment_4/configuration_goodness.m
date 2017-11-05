function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    nconfig = size(hidden_state, 2);
    total_energy = 0;
    for c=1:nconfig
        energy = transpose(hidden_state(:,c)) * rbm_w * visible_state(:,c);
        total_energy = total_energy + energy;
    end
    G = total_energy / nconfig;
end
