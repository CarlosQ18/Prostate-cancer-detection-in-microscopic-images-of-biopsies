clc; 
clearvars -global;
clear all;
%Base de Datos
rng("default");

Stats_path = fullfile(pwd,'LumenNuclei_GLCMstats.txt');
Stats_table = readtable(Stats_path);

% Cantidad de clases

cl = 4;

% Cantidad de caracteristicas
caract = size(Stats_table);
caract = caract(2) - 2;

% Separacion de Clases.
G0 = table2array(Stats_table(1:5000,2:end-1));
G3 = table2array(Stats_table(5001:10000,2:end-1));
G4 = table2array(Stats_table(10001:15000,2:end-1));
G5 = table2array(Stats_table(15001:20000,2:end-1));

DATOS = [G0;G3;G4;G5];

tp=size(DATOS);
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];
k=5;%127
[idx,w]=relieff(DATOS,Y,k);

save('relieff_LumenNuclei_GLCMstats.mat','idx','w')

idx = [109	110	45	122	61	152	60	151	4	145	59	103	102	129	25	80	123	165	120	113	166	168	67	3	50	49	144	163	149	124	150	87	19	131	89	57	140	51	38	52	139	18	138	81	127	66	153	108	128	142	107	121	62	71	53	78	130	161	91	112	154	147	133	105	141	48	29	72	82	73	159	134	36	143	101	167	17	88	5	90	104	119	24	26	114	86	100	95	116	158	98	137	68	92	164	148	99	39	115	106	126	65	79	63	117	30	31	111	96	16	2	160	44	58	21	40	15	162	8	84	132	94	93	83	77	97	125	70	12	156	7	155	14	32	85	135	1	75	10	69	9	136	146	118	64	157	37	23	6	41	35	28	74	22	13	42	33	27	76	20	55	11	34	54	43	56	47	46];