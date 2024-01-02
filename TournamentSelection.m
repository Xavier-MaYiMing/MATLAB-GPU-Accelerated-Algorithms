function index = TournamentSelection(K, N, fitness)
% K-tournament selection to find the indices of N solutions
    
    [~, rank] = sort(fitness);
    rnd = randi(length(fitness), K, N);
    [~, best] = min(rank(rnd), [], 1);
    index = rnd(best + (0 : N - 1) * K);
end
