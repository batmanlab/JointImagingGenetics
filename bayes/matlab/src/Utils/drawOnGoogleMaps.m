% this function plot 'fgImg' on a map acquired from google

function fusedImg= drawOnGoogleMaps(geoRange,fgImg,alpha)
    lat = geoRange(3:end) ;
    lon = geoRange(1:2) ;
    %figure
    %xlim(lon)
    %ylim(lat)
    [lonVec latVec map_img] = plot_google_map('maptype','roadmap',...
        'axis',[lon lat]) ;
    idx1 = (lonVec >= lon(1))  & (lonVec <=lon(2)) ;
    idx2 = (latVec >= lat(1))  & (latVec <=lat(2)) ;
    map_img_crop = map_img(idx1,idx2,:) ;
    fgImg_resize = imresize(fgImg,[size(map_img_crop,1) size(map_img_crop,2)])  ;
    fusedImg = FuseImages(map_img_crop, fgImg_resize, alpha) ;
    %imshow(fusedImg)
end