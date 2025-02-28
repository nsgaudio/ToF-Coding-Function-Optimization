
% %%%%%%%%%%%%%%%%%%%%%% PATCH ONLY VAL %%%%%%%%%%%%%%%%%%%%%%%%%%
% clearvars -except depths test;
% close all;
% 
% % This script generages square patches of specified dimension
% % the resuling patch set will be of dimension (N, D, D)
% 
% depths_new = depths(:, 10+1:end-22, 32+1:end-32);
% 
% % Converts from m to mm
% depths_new = 1000 * depths_new;
% 
% % Import depths.mat to the workspace
% N_in = length(depths_new(:,1,1));
% H_in = length(depths_new(1,:,1));
% W_in = length(depths_new(1,1,:));
% 
% % Select the patch dimension size and train, val, test split percentages.
% D = 64;
% train_percent = 0.90;
% val_percent = 0.05;
% test_percent = 0.05;
% 
% N_train = floor(N_in * train_percent);
% N_val =  N_train + floor(N_in * val_percent);
% N_test =  N_val + floor(N_in * test_percent);
% 
% patches_val = depths_new(N_train+1:N_val, :, :);
% patches_test = depths_new(N_val+1:N_test, :, :);
% 
% row_patches =  floor(H_in/D);
% col_patches = floor(W_in/D);
% N = N_train * row_patches * col_patches;
% 
% patches_train = single(zeros(N, D, D));
% 
% ind = 1;
% for n = 1:N_train
%     for r = 1:row_patches
%         for c = 1:col_patches
%             patches_train(ind, :, :) = depths_new(n, (r-1)*D+1:r*D, (c-1)*D+1:c*D);
%             ind = ind + 1; 
%         end
%     end
% end
% 
% patches_train = patches_train(randperm(length(patches_train(:, 1, 1))), :, :);
% filename = ['patches_', num2str(D), '.mat'];
% save(filename, 'patches_train', 'patches_val', 'patches_test');



% %%%%%%%%%%%%%%%%%%%%%% PATCH NONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clearvars -except depths test;
% close all;
% 
% % This script generages square patches of specified dimension
% % the resuling patch set will be of dimension (N, D, D)
% 
% depths_new = depths(:, 10+1:end-22, 32+1:end-32);
% 
% % Converts from m to mm
% depths_new = 1000 * depths_new;
% 
% % Import depths.mat to the workspace
% N_in = length(depths_new(:,1,1));
% H_in = length(depths_new(1,:,1));
% W_in = length(depths_new(1,1,:));
% 
% % Select the patch dimension size and train, val, test split percentages.
% D = 64;
% train_percent = 0.90;
% val_percent = 0.05;
% test_percent = 0.05;
% 
% N_train = floor(N_in * train_percent);
% N_val =  N_train + floor(N_in * val_percent);
% N_test =  N_val + floor(N_in * test_percent);
% 
% patches_val = depths_new(N_train+1:N_val, :, :);
% patches_test = depths_new(N_val+1:N_test, :, :);
% 
% patches_train = depths_new(1:N_train, :, :);
% save('patches_448_576.mat', 'patches_train', 'patches_val', 'patches_test')



%%%%%%%%%%%%%%%%%%%%%% PATCH ALL
clearvars -except depths test;
close all;

% This script generages square patches of specified dimension
% the resuling patch set will be of dimension (N, D, D)

depths_new = depths(:, 10+1:end-22, 32+1:end-32);

% Converts from m to mm
depths_new = 1000 * depths_new;

% Import depths.mat to the workspace
N_in = length(depths_new(:,1,1));
H_in = length(depths_new(1,:,1));
W_in = length(depths_new(1,1,:));

% Select the patch dimension size and train, val, test split percentages.
D = 64;
train_percent = 0.90;
val_percent = 0.05;
test_percent = 0.05;

N_train = floor(N_in * train_percent);
N_val =  N_train + floor(N_in * val_percent);
N_test =  N_val + floor(N_in * test_percent);

row_patches =  floor(H_in/D);
col_patches = floor(W_in/D);

patches_train = single(zeros(N_train * row_patches * col_patches, D, D));
patches_val = single(zeros((N_val - N_train) * row_patches * col_patches, D, D));
patches_test = single(zeros((N_test - N_val) * row_patches * col_patches, D, D));

ind = 1;
for n = 1:N_train
    for r = 1:row_patches
        for c = 1:col_patches
            patches_train(ind, :, :) = depths_new(n, (r-1)*D+1:r*D, (c-1)*D+1:c*D);
            ind = ind + 1; 
        end
    end
end

ind = 1;
for n = N_train+1:N_val
    for r = 1:row_patches
        for c = 1:col_patches
            patches_val(ind, :, :) = depths_new(n, (r-1)*D+1:r*D, (c-1)*D+1:c*D);
            ind = ind + 1; 
        end
    end
end

ind = 1;
for n = N_val+1:N_test
    for r = 1:row_patches
        for c = 1:col_patches
            patches_test(ind, :, :) = depths_new(n, (r-1)*D+1:r*D, (c-1)*D+1:c*D);
            ind = ind + 1; 
        end
    end
end

patches_train = patches_train(randperm(length(patches_train(:, 1, 1))), :, :);
% patches_val = patches_val(randperm(length(patches_val(:, 1, 1))), :, :);
% patches_test = patches_test(randperm(length(patches_test(:, 1, 1))), :, :);
filename = ['patches_train_test_val_', num2str(D), '.mat'];
save(filename, 'patches_train', 'patches_val', 'patches_test');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TEST SCRIPT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a test script that assumes any single RGB (3 channel) test 
% image.  It is used to ensure the patching works.  The test image is 
% split into patches (deconstructed) then reconstructed and shown.

% H_in = length(test(:,1,1));
% W_in = length(test(1,:,1));
% C = length(test(1,1,:));
% 
% % Select the patch dimension size
% D = 600;
% 
% row_patches =  floor(H_in/D);
% col_patches = floor(W_in/D);
% N = row_patches * col_patches;
% patches = single(zeros(N, D, D, C));
% 
% ind = 1;
% for r = 1:row_patches
%     for c = 1:col_patches
%         patches(ind, :, :, :) = test((r-1)*D+1:r*D, (c-1)*D+1:c*D, :);
%         ind = ind + 1;
%     end
% end
% 
% reconstructed = zeros(row_patches*D, col_patches*D, C);
% ind = 1;
% for r = 1:row_patches
%     for c = 1:col_patches
%         reconstructed((r-1)*D+1:r*D, (c-1)*D+1:c*D, :) = patches(ind, :, :, :);
%         ind = ind + 1;
%     end
% end
% 
% figure(1)
% imshow(uint8(squeeze(patches(1, :, :, :))))
% title('Example Patch');
% xlabel(['Dimension: ', num2str(D)]);
% 
% figure(2)
% subplot(1, 2, 1)
% imshow(uint8(reconstructed));
% title('Reconstructed');
% 
% subplot(1, 2, 2)
% imshow(test);
% title('Input');
