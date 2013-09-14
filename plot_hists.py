dat2010 = np.load('2010.npz')
cleans10 = dat2010['arr_0']
dps10 = dat2010['arr_1']

dat2013 = np.load('2013.npz')
cleans13 = dat2013['arr_0']
dps13 = dat2013['arr_1']

ok = cleans10[:, 4] > 0
p90 = cleans10[ok, 1]
figure(1)
clf()
hist(p90,  bins=np.arange(0, 3.0, 0.1))
title('Clean 2010')
savefig('clean2010.png')

ok = dps10[:, 4] > 0
p90 = dps10[ok, 1]
figure(2)
clf()
hist(p90,  bins=np.arange(0, 3.0, 0.1))
title('DP=True 2010')
savefig('dptrue2010.png')

ok = cleans13[:, 4] > 0
p90 = cleans13[ok, 1]
figure(3)
clf()
hist(p90, bins=np.arange(0, 3.0, 0.1))
title('Clean 2013')
savefig('clean2013.png')

ok = dps13[:, 4] > 0
p90 = dps13[ok, 1]
figure(4)
clf()
hist(p90,  bins=np.arange(0, 3.0, 0.1))
title('DP=True 2013')
savefig('dptrue2013.png')
