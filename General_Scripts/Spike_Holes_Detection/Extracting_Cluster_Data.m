function sp = Extracting_Cluster_Data(Path)
% This function simply extract clustered data
myKsDir = Path;
sp = loadKSdir(myKsDir);                                                    %% Kilosort function to obtain clusters

Clus=sp.cids(sp.cgs==2);                                                    %% Only the good ones
disp(['A total of ',num2str(length(Clus)),' good clusters'])