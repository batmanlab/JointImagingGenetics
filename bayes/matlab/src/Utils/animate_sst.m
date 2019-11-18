function animate_sst(lat, lon, data, time, delay, filename)
% ANIMATE_SST will animate a SST data set, and create an AVI if desired
%
%   ANIMATE_SST(LAT, LON, DATA, TIME) will animate a visualiztion of DATA
%   displayed with the axis labels LAT and LON for a given TIME vector.
%
%   LAT is a one dimensional vector of latitudes in degrees north.
%   LON is a one dimensional vectof of longitude in degrees east.
%   TIME is a one demsional vector of serial date numbers (whole and
%        fractional days since January 1 0000.
%   DATA is a three dimensional array where:
%       size(DATA) == [ length(LAT) length(LON) length(TIME) ]
%
%   ANIMATE_SST(LAT, LON, DATA, TIME, DELAY) will animate a visualiztion
%   pausing DELAY seconds between update (or as fast as MATLAB can
%   display if delay is very small).
%
%   DELAY is a scalar number of seconds (i.e. 0.5)
%
%   ANIMATE_SST(LAT, LON, DATA, TIME, DELAY, FILENAME) will animate a
%   visualiztion of DATA recording the output into an AVI file named
%   FILENAME.
%
%   FILENAME is a character array representing the filename of the AVI file
%            to create.  FILENAME may be relative or fully qualified.
%
%   See also AVIFILE,SET.

% 
% Copyright 2009-2010 The MathWorks, Inc.

%% Check inputs
if nargin < 5
    delay = 0.1;
end

if nargin < 6
    make_movie = false;
else
    make_movie = true;
end

if ndims(data) ~= 3
    error('Exepected 3D data');
end

%% Create Time Vector
dates = datestr(time, 'mmm-yyyy');

%% Get temperature extrema
min_val = floor(min(data(:)));
max_val =  ceil(max(data(:)));

CLim = [ min_val max_val];

%% Create figure
% use imagesc to display first frame
hImage = imagesc(lon, lat, data(:, :, 1));

% Correct Color Limits, and orientation
set(gca, 'CLim', CLim, 'YDir', 'normal');

% Turn on Colorbar
colorbar;

% label axes
xlabel('Logitude (Degrees East)' );
ylabel('Latitude (Degrees North)');

% set title
title(['Mean Surface Temperature for ' dates(1, :) ' in \circC']);

%% Make avi file if required
if make_movie
    aviobj = avifile(filename);
    Frame = getframe(gcf);
    aviobj = addframe(aviobj,Frame);
end

%% Loop through data

% For smaller data sets, the longest dimension might not be time, so length
% is not appropriate in this context.  Here we are getting the size in the
% third dimension (time) as our stopping point. 
endframe = size(data, 3);

for ii = 2:endframe;
    pause(delay);
    set(hImage, 'CData', data(:, :, ii));
    title(['Mean Surface Temperature for ' dates(ii, :) ' in \circC']);
    drawnow;
    
    if make_movie
        Frame = getframe(gcf);
        aviobj = addframe(aviobj,Frame);
    end
end

%% Close avi file if required
if make_movie
    aviobj = close(aviobj); %#ok
end

end