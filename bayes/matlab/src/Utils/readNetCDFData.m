%% Read a NetCDF File
% This script will step through the process of reading
% data from a NetCDF file.
%
% 
% Missing Data, as flagged in the NetCDF file, will be
% replaced with a NaN value (Not a Number).
%
% 
% Data is read from the file, and saved to the caller's
% workspace. If the NetCDF provides data attributes
% "scale_factor" and "add_offset" the script will first
% scale and then offset the data.
%
% For data that is 2 dimensional or higher, this script
% will rotate it on import so it becomes row-major (the
% MATLAB default) as apposed to column-major (the NetCDF
% default).  For 2D data this is taking the tranpose, for
% 3D or higher data, it is the equivalent of taking the
% transpose of each data plane along the third dimension.
%
%
% Data attributes are stored in a structure named with
% the same name as the data but ending in "_info".
%
% 
% After reading the data, a separate function is called to
% create an animated display of a subset of the data.  It
% creates an AVI video file of the animation and saves it
% in the present directory.
%
% NOTE: This is not a general solution for reading all 
%       NetCDF files.  This is an example based on a single
%       file and works for that file.  It likely will not
%       work for other formats.
%
% 
% NOAA_ERSST_V3 data provided by:
%     NOAA/OAR/ESRL PSD,
%     Boulder, Colorado, USA,
% from their Web site at:
%     http://www.cdc.noaa.gov/
% 
% The data represents NOAA Extended Reconstructed Sea
% Surface Temperature (SST) V3b.
% 
% Data URL: 
% http://www.cdc.noaa.gov/data/gridded/data.noaa.ersst.html
%
%
% More help on reading NetCDF files in MATLAB can be found in the
% documention:
% <http://www.mathworks.com/access/helpdesk/help/techdoc/import_export/f5-86568.html#brrbr9v-1 Online.>
% 
%
% Copyright 2009-2010 The MathWorks, Inc.
% 
% Input:
% output : 
%   outDataFrame : output data frame
% Example:
%   filename = '/home/kayhan/Data/OcceanData-1960-1990/MBT/ocldb1366908629.9925.MBT.nc';
%   [data,lat,lon,time] = readNetCDFData(filename)
function outDataFrame = readNetCDFData(filename)
%% Open file
%filename = '/home/kayhan/Data/OcceanData-1960-1990/MBT/ocldb1366563327.3233.MBT.nc';
ncid = netcdf.open(filename,'NC_NOWRITE');


%% Explore the Contents
[numdims,nvars,natts] = netcdf.inq(ncid);

%% Get Global File Information
for ii = 0:natts-1
    fieldname = netcdf.inqAttName(ncid, netcdf.getConstant('NC_GLOBAL'), ii);
    fileinfo.(fieldname) = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'), fieldname );
end

%disp(fileinfo);

%% Get Dimension Information
disp('Dimension Data');
fprintf('Name\t\tLength\n');
disp('-----------------');

% preallocate structure
dimension = repmat(struct('name', '', 'length', 0), numdims, 1);

for ii = 1:numdims
    [dimension(ii).name, dimension(ii).length] = netcdf.inqDim(ncid,ii-1); 

    % padding name for table layout
    padlength   = min(0, length(dimension(ii).name));
    name_padded = [dimension(ii).name repmat(' ', padlength+1)];

    fprintf('%s\t\t%d\n', name_padded, dimension(ii).length);
end

%% Get the Data
outDataFrame = [] ;
for ii = 1:nvars
    [name, ~, ~, natts] = netcdf.inqVar(ncid,ii-1);
    
    % Get Variable Attributes
    tmpstruct = struct();
    for jj = 1:natts
        fieldname = netcdf.inqAttName(ncid, ii-1, jj-1);
        tmpstruct.(fieldname) = netcdf.getAtt(ncid, ii-1, fieldname );
    end

    % Get raw data
    data = netcdf.getVar(ncid,ii-1);
    
    % Replace Missing Numbers (if necessary
    if (isfield(tmpstruct, 'missing_value') )
        data( data == tmpstruct.missing_value ) = NaN;
    end
    
    % Scale data (if necessary)
    if( isfield(tmpstruct, 'scale_factor') )
        data = double(data) * tmpstruct.scale_factor;
    end
    
    % Apply offset (if necessary)
    if( isfield(tmpstruct, 'add_offset') )
        data = data + tmpstruct.add_offset;
    end
    
    % Transpose data from column major to row major
    if( isnumeric(data) && ndims(data) > 2 )
        data = permute(data, [2 1 3:ndims(data)]);
    elseif ( isnumeric(data) && ndims(data) == 2 )
        data = data';
    end
    
    % store attribute and data with appropriate name
    varinfoname = [name '_info'];
    %assignin('caller', varinfoname, tmpstruct);
    %assignin('caller', name, data);
    outDataFrame.(name) = data ;
    outDataFrame.(varinfoname) = tmpstruct ;
end

%% Close File
netcdf.close(ncid);

%% Animate First 100 Samples and Create AVI

% Dates in MATLAB are stored as a whole and fractional number of days since
% January 1st 0000.  The NetCDF file read here uses an offset from 1800.
% To display the dates accurately we add the offset to the time vector to
% get days since the year 0000.
%date_offset = datenum('1800-1-1');

%animate_sst(lat, lon, sst(:, :, 1:100), time(1:100)+date_offset, 0.2, 'test_sst.avi');

%% Clean Up temporary variables
%clear ndims nvars natts ii jj tmpstruct idx ncid filename
%clear fieldname name vartype dimids varinfoname data date_offset

end
