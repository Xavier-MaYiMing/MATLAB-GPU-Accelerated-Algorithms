function Obj = Evaluation(Pop)
% Shifted sphere function in CEC2008
% -100 <= x <= 100
    CallStack = dbstack('-completenames');
    load(fullfile(fileparts(CallStack(1).file), 'CEC2008.mat'), 'Data');
    O = Data{1};
    Z = Pop - repmat(O(1 : size(Pop, 2)), size(Pop, 1), 1);
    Obj = sum(Z .^ 2, 2);
end
