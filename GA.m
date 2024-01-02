function GA(N, maxFE, lb, ub)
% Genetic algorithm with simulated binary crossover (SBX) and polynomial mutation (PM)
   
    %% Step 1. Initilization
    [proC, disC, proM, disM] = deal(1, 20, 1, 20);
    Lower = repmat(lb, N, 1);
    Upper = repmat(ub, N, 1);
    Pop = unifrnd(Lower, Upper);
    Obj = Evaluation(Pop);
    FE = N;
    Gbest = [min(Obj)];

    %% Step 2. Optimization
    while FE <= maxFE
        MatingPool = TournamentSelection(2, N, Obj);
        Off = OperatorGA(Pop(MatingPool, :), Lower, Upper, proC, disC, proM, disM);
        Pop = [Pop; Off];
        Obj = [Obj; Evaluation(Off)];
        [~, rank] = sort(Obj);
        Pop = Pop(rank(1 : N), :);
        Obj = Obj(rank(1 : N));
        Gbest = [Gbest, Obj(1)];
        FE = FE + N;
    end

    %% Step 3. Output
    disp('The global best solution is: ');
    disp(Pop(1, :));
    disp('The global best result is: ');
    disp(Gbest(end));
    plot(1 : length(Gbest), Gbest, 'LineWidth', 2);
    xlabel('Iteration number');
    ylabel('Global best')
    title('GA training curve')
end