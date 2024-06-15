%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%Plot iterations%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filePath = '~/Documents/Computation_Comparisons';
modelFolder = 'model524';

startIter = 120;
endIter   = 129;
iterNums  = linspace(startIter, endIter, (endIter - startIter) + 1 );

varname    = 'chi1'; varnameLab = 'chi1';
w          = load([filePath '/' modelFolder '/w.dat']);

varMatrix = zeros(size(w,1), endIter - startIter);

for i = 1:(endIter - startIter)
    varMatrix(:,i) = load([filePath '/' modelFolder '/' varname '_' num2str(iterNums(i)) '.dat']);
    figure('visible', 'off')
    plot(w, varMatrix(:,i)); title([varnameLab '; iter: ' num2str(iterNums(i))]); xlabel('w')
    %ylim([-2.2 -1.2])
    saveas(gcf,[filePath '/' modelFolder '/iteration_charts/'  varname '_' num2str(iterNums(i))  '.png'])
end

varMatrix = zeros(size(w,1), endIter - startIter);

for i = 1:(endIter - startIter)
    varMatrix(:,i) = load([filePath '/' modelFolder '/' 'dzeta_h_dx_0' '_' num2str(iterNums(i)) '.dat']) ... 
        - load([filePath '/' modelFolder '/' 'dzeta_e_dx_0' '_' num2str(iterNums(i)) '.dat']);
    figure('visible', 'off')
    plot(w, varMatrix(:,i)); title(['(dzetaH / dw - dzetaE/ dw)' '; iter: ' num2str(iterNums(i))]); xlabel('w')
    xlim([0.7 1])
    saveas(gcf,[filePath '/' modelFolder '/iteration_charts/'  'dzeta_e_dw_dzeta_h_dw' '_' num2str(iterNums(i))  '.png'])
end

%%%%%

zetaH_219 = load([filePath '/' modelFolder '/' 'zeta_h' '_' num2str(284) '.dat']);
chi_219 = load([filePath '/' modelFolder '/' 'chi' '_' num2str(284) '.dat']);
dzetaH_219 = load([filePath '/' modelFolder '/' 'dzeta_h_dx_0' '_' num2str(284) '.dat']);
dzetaE_219 = load([filePath '/' modelFolder '/' 'dzeta_e_dx_0' '_' num2str(284) '.dat']);

figure();
plot(w, [chi_219 dzetaH_219 (1 + dzetaE_219)])
ylim([0 1.2])


dzetaH_219_dw = zeros(size(zetaH_219));
dzetaH_219_dw(2:end-1) = (zetaH_219(3:end) - zetaH_219(1:end-2) ) / (2 * (w(3) - w(2) ) );
dzetaH_219_dw(1) = (zetaH_219(2) - zetaH_219(1)) / (w(3) - w(2));
dzetaH_219_dw(end) = (zetaH_219(end) - zetaH_219(end-1)) / (w(3) - w(2));
