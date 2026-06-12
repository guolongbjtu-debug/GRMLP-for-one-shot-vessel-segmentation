% ==================================代码说明================================
%{  
    时间：GL 于 20211207
    地点：BJTU实验室
    功能：读取txt文档的后几行数据，去掉空行，转化为double矩阵;遇到非数字就停止读取

    参数说明：
        input paras
            file_name：  文件名，如‘result.txt’   => string
            N：          读取倒数几行数据
        output paras
            out：        读取的数据     => N*？ double mat
        function paras
            file_id      文件的指针
            row_num      row number of the txt
            row_inclusion   每一行转化为一个char，放在一个cell里面                => cell
            N_count      已经读取倒数N行的有效数据
%}
%   -----------------------Code debugging area-----------------------------
    % clear,clc
    % file_name = 'result.txt';
    % N = 2;
    % out = users_read_txt_row(file_name,N)
    % 
    
%% =========================================================================    
%     
function out = users_read_txt_last_rows(file_name,N)
    % 读取每一行的char型文本，存储在row_inclusion 的cell文件中
        file_id = fopen(file_name);         % 文件的指针
        row_num = 0;
            while ~feof(file_id)            % 逐行读取 ~feof(file_name) 表示文件指针到达文件末尾时 该表达式值为“假”；否则为“真”,"真"的话就执行循环里面的
                row_num = row_num + 1;
                row_inclusion{row_num} = fgetl(file_id);       
                                            % 从文件中读取一行数据(字符串的形式)，并去掉行末的换行符。
            end
        fclose(file_id);                    % 重置指针


    % 读取后几行的数字，并排除空行
        N_count = 0;
        for row_count = row_num : -1 : 1                          % 倒序判断每一行
           if row_inclusion(row_count) ~= ""  &&  N_count < N     % 如果不是空行 并且 没记录到N行有效数据
               try
                N_count = N_count + 1;
                out(N_count,:) = str2num(row_inclusion{row_count});
               catch                                              % 遇到非数字就break for循环
                 break  
               end
           end
        end
end