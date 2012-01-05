function print_S(fid,S)
for i=1:length(S)
        fprintf(fid,'%d',S(i));
        if mod(i,32)==0
            fprintf(fid,'\n');
        end
end
