function msk = WeakTextureMask(img, th, patchsize)
if( ~exist('patchsize', 'var') )
    patchsize = 7;
end
kh = [-1/2,0,1/2];
imgh = imfilter(img,kh,'replicate');
imgh = imgh(:,2:size(imgh,2)-1,:);
imgh = imgh .* imgh;
kv = kh';
imgv = imfilter(img,kv,'replicate');
imgv = imgv(2:size(imgv,1)-1,:,:);
imgv = imgv .* imgv;
s = size(img);
msk = zeros(s);
for cha=1:s(3)
	m = im2col(img(:,:,cha),[patchsize patchsize]);
	m = zeros(size(m));
	Xh = im2col(imgh(:,:,cha),[patchsize patchsize-2]);
	Xv = im2col(imgv(:,:,cha),[patchsize-2 patchsize]);
    
	Xtr = sum(vertcat(Xh,Xv));
	
	p = (Xtr<th(cha));
	ind = 1;
	for col=1:s(2)-patchsize+1
		for row=1:s(1)-patchsize+1
			if( p(ind) > 0 )
				msk(row:row+patchsize-1, col:col+patchsize-1, cha) = 1;
			end
			ind = ind + 1;
		end
    end
end
end
