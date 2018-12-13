function[min_x,min_y] = Draw_ROC(decision,datalabel)
    TN = 0;
    FP = 0;
    FN = 0;
    TP = 0;
    for i = 1:length(decision)
       if decision == 0 & datalable == 0
           TN = TN + 1;
       elseif decision == 1 & datalable == 0
           FP = FP + 1;
       elseif decision == 0 & datalable == 1
           FN = FN + 1;
       else
           TP = TP + 1;
       end
    end
    pos_num = sum(datalabel == 1);
    neg_num = sum(datalabel == 0);
    [pre,Index]=sort(decision);
    datalabel=datalabel(Index);
    m = length(decision);
    x=zeros(m+1,1);
    y=zeros(m+1,1);
    x(1)=1;y(1)=1;
    auc = 0;
    max_Youden = 0;
    min_x = 0;
    min_y = 0;
    for i=2:m
        TP=sum(datalabel(i:m)==1);
        FP=sum(datalabel(i:m)==0);
        x(i)=FP/neg_num;
        y(i)=TP/pos_num;
        d = y(i) - x(i);
        if d > max_Youden
           max_Youden = d;
           min_x = x(i);
           min_y = y(i);
        end
        auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
    end
    x(m+1)=0;y(m+1)=0;
    auc=auc+y(m)*x(m)/2;
    plot(x,y);
    hold on
    xlabel('False positive rate');
    ylabel('True positive rate');
    %legend("ROC for Minimum expected loss","ROC for guess");
    title('ROC curve');
end