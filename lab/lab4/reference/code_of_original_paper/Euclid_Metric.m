function my_matrix = Euclid_Metric(TA_matrix)
number_of_column = size(TA_matrix,1);
my_matrix = zeros(number_of_column^2,3);
count = 0;
for i = 1:number_of_column
    for j = 1:number_of_column
        count = count + 1;
        my_matrix(count,1) = i;
        my_matrix(count,2) = j;
        my_matrix(count,3) = ( (TA_matrix(i,1)-TA_matrix(j,1))^2 + (TA_matrix(i,2)-TA_matrix(j,2))^2 )^(1/2) ;
    end
end
end