using Test
using SCYFI
using LinearAlgebra
using Plots

#randomly generate sys
# M=2
# H=10
# A=randn(M)
# W₁=randn(M,H)
# #W₁ =W₁-Diagonal(W₁)
# W₂=randn(H,M)
# #W₂ =W₂-diagm(diag(W₂))
# h₁=randn(M)
# h₂=randn(H)
# p=plot(xlims=(-100,100),ylims=(-100,100),legend=false)
# for k=1:200
#     traj= get_latent_time_series(10000,A,W₁,W₂,h₁,h₂,M, z_0=randn(M), is_clipped=true)
#     traj=reshape(collect(Iterators.flatten(traj)), (length(traj[1]),length(traj)))
#     #println(size(traj[2][2]))
#     #println(traj[1,end-10:end])
#     Plots.plot!(p,traj[1,end-100:end],traj[2,end-100:end],label="k=$k",marker=:circle)
# end
# display(p)

#Plots.plot(p,traj[1,begin:20],traj[2,begin:20])
#plott the first vs the second entries in trajfor all entries
#reshape(Array(traj),10000,2)
#println(traj[1,:])
#reshape(collect(Iterators.flatten(a)), (length(a[1]),length(a)))
#plot!(p,xlims=(-5,5),ylims=(-5,5),legend=false)
#cycles, eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,2,outer_loop_iterations=100,inner_loop_iterations=250,PLRNN=ClippedShallowPLRNN())
#println(cycles)#[1][3][1])
###Plots.scatter!(p,[cycles[2][1][2][1]],[cycles[2][1][2][2]],label="cycle",marker=:cross,markersize=10,color="red")#,xlims=(-10,10),ylims=(-10,10),color=:red)

#println(A)
#println(W₁)
#println(W₂)
#println(h₁)
#println(h₂)

function test_finding_1_cycle_M2_H10_clipped()
    # define variables for GT sys with 1 cycle if
    M=2
    A=[0.5917591190018257, -0.2670556599714421]
    W₁=[1.068513167265907 1.1348797864408178 0.08228632519950883 -0.9427934205409179 0.642705978131473 2.6759515297389487 0.7060779536633661 0.7403096229220267 0.21090714408336952 1.7996677685472369; -1.4279184362686048 0.6651831504837336 0.05794900129098961 0.6621966950807933 -0.7228533075320732 -0.44469319962883574 -0.045486947312179594 0.9271984361948209 0.831193439154738 1.4045447403219602]
    W₂=[1.2557874562489009 -0.53340159623648; 0.18028758395331218 -0.37567264544875484; 0.5251380318002776 1.2182973412833118; 1.107554538603655 0.19665533183681763; 0.24040135957738534 0.15208824606722526; -1.4240039952629835 -1.3505046047814038; 0.051864276456367674 -0.897424734041841; -0.4695192965033343 -1.3708025903165928; 1.8783433960135856 -0.10994708797849907; -0.8724152972898058 -0.6382212185647201]
    h₁=[-0.03887946906504432, -0.5638441067359811]
    h₂=[0.583723591983208, 1.6594475075515556, 1.0987707695942321, -1.1086657549577021, 0.5102632745463866, -0.7623718435787896, 2.0653625210236695, -0.6966999800075485, 1.6483574105782965, -0.4405880221910508]
    traj= get_latent_time_series(10000,A,W₁,W₂,h₁,h₂,M, z_0=randn(M), is_clipped=true)
    traj=reshape(collect(Iterators.flatten(traj)), (length(traj[1]),length(traj)))
    FPs,eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,1,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ClippedShallowPLRNN(),get_pool_from_traj=true)
    @test round.(FPs[1][1][1],digits=3)==round.(traj[:,end],digits=3)	
    @test length(FPs[1][1]) == 1
end
#test_finding_1_cycle_M2_H10_clipped()

function test_finding_1_cycle_M2_H10_clipped2()
    M=2
    A=[-0.06419553876846194, -0.2622863003398774]
    W₁=[1.1184264251418332 1.091258176637432 1.007314936321882 0.5194199785778972 1.2640699367247659 0.7074665102632437 0.7098906929124482 0.9436473394062519 0.713883619256823 -0.0687002530494658; 0.321988916180829 -0.4687365588406984 -1.3131903999387478 -0.8264152224966445 0.12946849779042788 1.5877682970526805 -0.5418050507617767 1.4063256304876521 0.6091856686836092 -0.620096915093448]
    W₂=[-0.933161682992203 0.3612837459075444; 1.0641225255009603 -0.516641883593588; 0.5300369254468582 0.14438503703551506; 0.05595392000357566 -0.8215896592379517; -0.975132243233939 0.07407576083196643; 0.6763710517948548 -0.5433063167809091; 0.7373835106735419 -2.4266517766088724; 0.5681912879818395 0.5635653215832376; -0.9629690378364417 -0.4045912014913528; -1.718270984416333 0.5931857971585359]
    h₁=[-1.3014888652812306, 0.8599606573606556]
    h₂=[-1.6125719302354802, 0.26956168340118447, -0.13704234891581388, 0.27419730015064353, 1.0272075596110417, 0.4939306531634047, 0.8145865783296456, 1.7410992960816623, -0.5513836306147909, 0.6351910001883486]
    traj= get_latent_time_series(10000,A,W₁,W₂,h₁,h₂,M, z_0=randn(M), is_clipped=true)
    traj=reshape(collect(Iterators.flatten(traj)), (length(traj[1]),length(traj)))

    FPs,eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,1,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ClippedShallowPLRNN())
    @test round.(FPs[1][1][1],digits=3)==round.(traj[:,end],digits=3)	
    @test length(FPs[1][1]) == 1
end
#test_finding_1_cycle_M2_H10_clipped2()

#println(FPs[1][1][1])
#round.(FPs[1][1][1],digits=3)==round.(traj[:,end],digits=3)
function test_finding_2_cycle_M2_H10_clipped_val()	
    M=2
    A=[-0.387934780653102, 0.5123045746686165]
    W₁=[-0.9572042035845127 -0.1830689421517754 0.612724219548259 0.5072123624933069 -2.93833010824187 -0.7166910570852257 0.09483144567532248 0.3275011463581499 -0.596729272294966 0.21833363030454014; 0.11252030787955322 0.08361706357908519 -0.1883799922944963 0.11798518366934412 2.214442358271848 0.5048710768032438 0.6530100515570298 0.3616554430766955 0.17756613771012567 0.1607826194696379]
    W₂=[0.03312283756823697 1.8839667964477298; -1.8645741228197934 -0.343038769506633; 0.32178514086049653 -0.6546290047717681; -1.7138058159480039 -2.170483207993156; 1.0357378202180176 2.679592104175041; 0.8326308548714895 0.60488813592394; 0.11291096347254374 -0.12185732820579509; 0.8078735346646678 1.2608387934459058; 0.6370601193078352 0.8861459726379051; -0.665435960662722 0.08213441509969478]
    h₁=[-0.577277965780346, -2.009240385851437]
    h₂=[0.8614300380448116, -1.2928727603452919, 0.04562658977737302, -0.24653003991713363, -0.14907612939827858, 1.6511523265499823, 1.3836150637812514, -0.11713937751742667, 0.8566129911695514, 1.5013701728321438]
    traj= get_latent_time_series(10000,A,W₁,W₂,h₁,h₂,M, z_0=randn(M), is_clipped=true)
    traj=reshape(collect(Iterators.flatten(traj)), (length(traj[1]),length(traj)))

    FPs,eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,2,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ClippedShallowPLRNN())
    @test length(FPs[2]) == 1
    @test round.(FPs[2][1][1][1],digits=3) ∈ round.(traj[:,end-1:end],digits=3)	

end

function test_finding_10_cycle_M2_H10_clipped()	
    A=[0.60122037, 0.8135468, 0.7626706]
    W₁=[0.11067849 -0.1302824 0.25012589 -0.22298877 0.15806502 -0.34588745 -0.06457776 -0.09123066 -0.23283756 0.16113971; -0.75709593 -0.62422514 -0.20954604 -0.66458714 0.28777176 -0.13910763 0.013174338 0.64183784 -0.5570398 0.24596561; 0.2582224 0.027766248 0.5165033 -0.30535492 0.28937963 -0.59222066 -0.08215613 -0.16016932 -0.41183522 0.34899935]
    W₂=[-0.243685 -0.3262217 -0.525551; 0.23788112 0.111583985 0.48059195; -0.0061925445 0.5009318 0.0021386577; -0.2870248 0.21104889 -0.55250746; -0.067662284 0.2247282 -0.19935167; -0.048656207 -0.45318457 0.027064549; -0.0005488223 0.08707904 0.00031374826; 0.219448 0.10352938 0.40836236; 0.18324012 -0.4742444 0.3508795; 0.06861267 -0.24380997 0.19178945]
    h₁=[-0.04333596, -0.09287525, -0.08091329]
    h₂=[0.45786792, -0.33195782, -0.3534318, 0.38365522, -0.31813976, -0.5264057, 0.061465904, 0.47354963, -0.3893013, 0.3268898]
    
    FPs,eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,10,outer_loop_iterations=30,inner_loop_iterations=200,PLRNN=ClippedShallowPLRNN())
    @test length(FPs[10]) > 0
    #Visualization (needs BPTT)
    # m,O=load_model("../bptt-julia/Results/test_clipped/clipped-shallow-new-cycle_example_data/data_cycle_10_0ellipses.npy-alpha_0.4-M_3-H_10-reg_1.0e-5-epochs_15000-noise_0.01-batchsize_16/001/checkpoints/model_10000.bson")
    # cycles, eigenvals =find_cycles(m.A, m.W₁,m.W₂, m.h₁,m.h₂,10,outer_loop_iterations=100,inner_loop_iterations=500,PLRNN=ClippedShallowPLRNN())
    # println(m.A)
    # println(m.W₁)
    # println(m.W₂)
    # println(m.h₁)
    # println(m.h₂)
    # using Plots

    # traj = generate(m,O,randn(2),10000)

    # p = plot(traj[end-100:end,1],traj[end-100:end,2],legend=false)

    # cycles[10][1]
    # cyc=reshape(collect(Iterators.flatten(cycles[10][1])), (length(cycles[10][1][1]),length(cycles[10][1])))
    # plot!(p,cyc[1,:],cyc[2,:],linewidth=3,linestyle=:dash,marker=:circle,markersize=3,color="red")
end
	
#test_finding_1_cycle_M2_H10_clipped()

function test_finding_20_cycle_M2_H10_clipped()	
    A=[0.9338895, 0.8158928, 0.64717215]
    W₁=[-0.08257616 -0.06510486 -0.3011948 -0.01509484 -0.24916363 0.20184407 0.18710229 -0.26356283 0.20855018 -0.26561627; 0.4785514 -0.5583767 0.31031066 -0.415861 -0.38136455 -0.13172355 0.52841985 -0.058546904 -0.4165734 0.0913285; -0.02599251 0.032929726 0.035843737 0.09668805 -0.030407568 -0.013791576 0.032235596 0.04951459 0.09209862 0.046291262]
    W₂=[-0.40293103 -0.14325522 0.054672144; 0.48560253 0.007107038 0.12721366; -0.23135893 -0.23621486 0.05824172; -0.3275432 -0.14798847 -0.09055013; 0.3009528 -0.07327408 -0.17328814; 0.046891127 0.1438347 -0.08094727; 0.4461682 -0.11263572 -0.15933159; -0.15238671 0.20166379 0.009564617; -0.3536613 -0.21320055 0.0042185816; -0.09381822 0.17765957 -0.13131015]
    h₁=[-0.066404805, 0.043454356, 0.28484693]
    h₂=[-0.225078, -0.25648823, -0.2741254, 0.21262659, -0.28708217, -0.17037255, 0.28027406, 0.10760485, 0.23348168, 0.15340137]
    
    FPs,eigenvals =find_cycles(A, W₁,W₂, h₁,h₂,20,outer_loop_iterations=30,inner_loop_iterations=200,PLRNN=ClippedShallowPLRNN())
    @test length(FPs[20]) > 0
    @test length(FPs[10]) == 0
    #visualization (needs BPTT)
    # traj = generate(m,O,randn(2),10000)

    # p = plot(traj[end-100:end,1],traj[end-100:end,2],legend=false)

    # cycles[20]
    # cyc=reshape(collect(Iterators.flatten(cycles[20][2])), (length(cycles[20][2][1]),length(cycles[20][2])))
    # plot!(p,cyc[1,:],cyc[2,:],linewidth=3,linestyle=:dash,marker=:circle,markersize=3,color="red")

end
test_finding_20_cycle_M2_H10_clipped()	
# FPs
# traj[:,end-2:end]
# FPs
# round.(FPs[2][1][1],digits=3)
# round.(traj[end,:],digits=3)	
# round.(FPs[2][1][1],digits=3)==round.(traj[:,end],digits=3)	
