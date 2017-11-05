min_loss = 1e20;
for lrate = (35:100) ./ 1000
    lrate
    current_loss = a4_main_edit(300, 0.02, lrate, 1000);
    if current_loss < min_loss
        min_loss = current_loss;
        disp('-------------------------------> Found a better learning rate <-------------------------------');
    end
    disp('***********************************************');
end
