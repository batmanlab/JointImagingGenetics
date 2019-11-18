fn='../data/crimeData/crime3yr_data.mat' ;
[~,dataCell] = convertCrime2Grid(fn) ;
numTimes = length(dataCell) ;
lin_betaCell = cell(numTimes-5,1) ;
lin_pCell = cell(numTimes-5,1) ; 
quad_betaCell = cell(numTimes-5,1) ;
quad_pCell = cell(numTimes-5,1) ;

for tCnt1=1:numTimes-5
    lin_betaCell{tCnt1} = zeros([size(dataCell{1}) 2]) ;
    lin_pCell{tCnt1} = ones(size(dataCell{1})) ;
    quad_betaCell{tCnt1} = zeros([size(dataCell{1}) 3]) ;
    quad_pCell{tCnt1} = ones(size(dataCell{1})) ;
    for rowCnt=1:size(dataCell{1},1)
        for colCnt=1:size(dataCell{1},2)
            y = [] ;
            t = [] ;
            for tCnt2=tCnt1:tCnt1+4
                y = [y; dataCell{tCnt2}(rowCnt,colCnt)] ;
                t = [t; tCnt2] ;
            end    
            t = t - tCnt1 ;
            stat1 = regstats(y,t,'linear') ;
            stat2 = regstats(y,t,'quad') ;
            lin_betaCell{tCnt1}(rowCnt,colCnt,:) = stat1.beta ;
            lin_pCell{tCnt1}(rowCnt,colCnt) = stat1.fstat.pval ;
            quad_betaCell{tCnt1}(rowCnt,colCnt,:) = stat2.beta ;
            quad_pCell{tCnt1}(rowCnt,colCnt) = stat2.fstat.pval ;
        end
    end
end
fn2 = '../data/crimeData/crime3yr_changeRate.mat' ;
save(fn2,'lin_betaCell','lin_pCell','quad_betaCell','quad_pCell')