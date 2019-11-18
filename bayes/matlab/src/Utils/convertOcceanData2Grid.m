% this function converts the lon, lat, data from a vector format into 
% grid format.
% inputs:
%     lon: N x 1 vector containing lon coordinate
%     lat: N x 1 vector containing lat coordinate
%     data: N x 1 vector containing data 
%     time: N x 1 vector containing time
%     opt : options structure, it should contain the following data
%         .numLonGrid : number of Lon grid
%         .numLatGrid : number of Lat grid
%         .numTimeGrid : number of time grid
function  frameMean = convertOcceanData2Grid(lon,lat,data,time,opt)
          % basic clean up to make sure that all time points are positive
          idx = find(time > 0 ) ;
          time = time(idx) ;
          lon = lon(idx) ;  % x-axis
          lat = lat(idx) ;  % y-axis
          data = data(idx) ;
          
          minLat = min(lat) ;
          maxLat = max(lat) ;
          minLon = min(lon) ;
          maxLon = max(lon) ;
          minTime = min(time) ;
          maxTime = max(time) ;
          
          %[LON,LAT] = meshgrid(linspace(minLon,maxLon+eps,opt.numLonGrid+1),...
          %                     linspace(minLat,maxLat+eps,opt.numLatGrid+1)) ;
                           
          lonGrid = linspace(minLon,maxLon+eps,opt.numLonGrid+1) ;
          latGrid = linspace(minLat,maxLat+eps,opt.numLatGrid+1) ; 
          timeGrid = linspace(minTime,maxTime+eps,opt.numTimeGrid+1) ;                 
          
          frameMean = zeros(opt.numLatGrid,opt.numLonGrid,opt.numTimeGrid) ;
          for timeCnt=1:opt.numTimeGrid
              timeCnt
              data1 = data(time >= timeGrid(timeCnt) &  time < timeGrid(timeCnt+1) ) ;
              lon1 = lon(time >= timeGrid(timeCnt) &  time < timeGrid(timeCnt+1) ) ;
              lat1 = lat(time >= timeGrid(timeCnt) &  time < timeGrid(timeCnt+1) ) ;
              avg = mean(data1) ;
              for lonCnt=1:opt.numLonGrid
                data2 = data1(lon1 >= lonGrid(lonCnt) & lon1 < lonGrid(lonCnt+1)) ;
                lat2 = lat1(lon1 >= lonGrid(lonCnt) & lon1 < lonGrid(lonCnt+1)) ;
                for latCnt=1:opt.numLatGrid
                      data3 = data2( lat2 >= latGrid(latCnt) & lat2 < latGrid(latCnt+1) ) ;
                      %idx = time >= timeGrid(timeCnt) &  time < timeGrid(timeCnt+1) & ...
                      %      lon >= lonGrid(lonCnt) & lon < lonGrid(lonCnt+1) & ...
                      %      lat >= latGrid(latCnt) & lat < latGrid(latCnt+1) ;
                      if ~isempty(data3)
                          frameMean(latCnt,lonCnt,timeCnt) = mean(data3-avg) ;
                      else
                          frameMean(latCnt,lonCnt,timeCnt) = 0 ;
                      end
                 end
              end
          end
          
end