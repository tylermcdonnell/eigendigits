function [ ] = displayimages( images )
%DISPLAYIMAGE Displays a number of square images in subplots.
%   Takes an [N x N x M] array where N x N is the dimension of an image
%   and M is the number of images and display the M images in subplots for
%   convenient viewing.

[~, n] = size(images);
for i = 1:n
    subplot(n/5,5,i), subimage(toimage(images(:,i)));
    set(gca, 'XTickLabel', '');
    set(gca, 'YTickLabel', '');
end

end

