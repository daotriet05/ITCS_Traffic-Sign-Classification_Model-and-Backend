beta = 1e-2
gamma = 1e-1
mix_ratio = 0.5
feature_channel = 6
# show one picture for example in the last batch
show_pic = 0
show_batch = 0
save_epoch = 2 # save log images per save_epoch


# Define model or load model
net = get_model(args.arch, class_train, class_test, feature_channel)
net.cuda()
if args.resume is not None:
  net = torch.load(args.resume)


# Define loss fucntions
match_loss = nn.MSELoss(reduction='mean')
recon_loss = nn.BCELoss()

def PIDA_loss(feature, target):
    tg_unique = torch.unique(target)
    pida_loss = 0
    for tg in tg_unique:
      feature_split = feature[target==tg,:,:,:]
      mean_feature = torch.mean(feature_split, 0).unsqueeze(0)
      mean_feature_rep = mean_feature.repeat(feature_split.shape[0], 1, 1, 1)
      pida_loss += match_loss(feature_split, mean_feature_rep)
    return pida_loss

def loss_class_func(out, target, out_sup=None, target_sup=None):
    CE = F.cross_entropy(out, target)
    CE_sup = 0
    if out_sup is not None:
        CE_sup = F.cross_entropy(out_sup, target_sup)
    return CE + CE_sup
    
def loss_match_func(feat_sem, temp_sem):
    MS = match_loss(feat_sem, temp_sem)
    return MS
    
def loss_recon_func(recon_feat_sem, recon_temp_sem, template, recon_temp_sup=None, template_sup=None):
    RE = recon_loss(recon_feat_sem, template) + recon_loss(recon_temp_sem, template)
    if recon_temp_sup is not None:
        recon_sup = recon_loss(recon_temp_sup, template_sup)
        RE += recon_sup
    return RE
    
def loss_illu_func(feat_illu, target):
    pida_illu = PIDA_loss(feat_illu, target)
    return -pida_illu


# Construct optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

tr_class = tr_loader.tr_class
te_class = te_loader.te_class

def train(e):
  net.train()
  
  print('start train epoch: %d'%e)
  
  corr = 0
  all = 0
  feat_corr = 0
  feat_all = 0
  for i, (input, target, template) in enumerate(trainloader):
    optimizer.zero_grad()
    target = torch.squeeze(target)
    input, template = input.cuda(async=True), template.cuda(async=True)

    # extract
    feat_sem, feat_illu, feat_nowarp = net.extract(input, is_warping=True)
    temp_sem, _, _ = net.extract(template, is_warping=False)
    
    # decode
    recon_feat_sem = net.decode(feat_sem)
    recon_temp_sem = net.decode(temp_sem)

    # combine features via mixup or concat (see supplementary)
    #feature_exc = mix_ratio * feat_sem + (1-mix_ratio) * feat_illu
    feature_exc = torch.cat((feat_sem, feat_illu), 1)
    target_exc = target
    
    
    choose_tr = 1
    for choice in range(choose_tr):
      for j, tg in enumerate(target):
        cls = tr_class[random.randint(0, len(tr_class)-1)]
        template_new, _ = tr_loader.load_template([cls], augmentations=None)
        template_new = template_new.cuda(async=True)
        tg_new = torch.Tensor([cls]).long()
        temp_sem_new, _, _ = net.extract(template_new, is_warping=False)
        # combine features via mixup or concat
        #feature_new = mix_ratio * temp_sem_new + (1-mix_ratio) * (feat_illu[j]).unsqueeze(0)
        feature_new = torch.cat((temp_sem_new, (feat_illu[j]).unsqueeze(0)), 1)
        feature_exc = torch.cat((feature_exc, feature_new), 0)
        target_exc = torch.cat((target_exc, tg_new), 0)

        
    choose_sup = 1
    count_new = 0      
    recon_temp_sup = torch.zeros(choose_sup*input.shape[0],3,input.shape[2],input.shape[3]).cuda(async=True)
    temp_trans_sup = torch.zeros(choose_sup*input.shape[0],3,input.shape[2],input.shape[3]).cuda(async=True)
    template_sup = torch.zeros(choose_sup*input.shape[0],3,input.shape[2],input.shape[3]).cuda(async=True)
    feat_sem_sup = torch.zeros(choose_sup*input.shape[0],int(feature_channel/2),input.shape[2],input.shape[3]).cuda(async=True)
    temp_sem_sup = torch.zeros(choose_sup*input.shape[0],int(feature_channel/2),input.shape[2],input.shape[3]).cuda(async=True)
    target_sup = torch.zeros(choose_sup*input.shape[0]).long()
    feature_sup = torch.zeros(choose_sup*input.shape[0],int(feature_channel),input.shape[2],input.shape[3]).cuda(async=True)
    
    for choice in range(choose_sup):
      for j, tg in enumerate(target):
        cls = te_class[random.randint(0, len(te_class)-1)]
        temp_trans_new, template_new = te_loader.load_template([cls], augmentations=template_trans_support)
        temp_trans_new = temp_trans_new.cuda(async=True)
        template_new = template_new.cuda(async=True)
        tg_new = torch.Tensor([cls]).long()
        temp_trans_sem_new, _, _ = net.extract(temp_trans_new, is_warping=True)
        temp_sem_new, _, _ = net.extract(template_new, is_warping=False)
        # combine features via mixup or concat
        #feature_new = mix_ratio * temp_trans_sem_new + (1-mix_ratio) * (feat_illu[j]).unsqueeze(0)
        feature_new = torch.cat((temp_trans_sem_new, (feat_illu[j]).unsqueeze(0)), 1)
        
        recon_temp_sup[count_new,:,:,:] = net.decode(temp_trans_sem_new)
        temp_trans_sup[count_new,:,:,:] = temp_trans_new
        template_sup[count_new,:,:,:] = template_new
        feat_sem_sup[count_new,:,:,:] = temp_trans_sem_new
        temp_sem_sup[count_new,:,:,:] = temp_sem_new
        
        target_sup[count_new] = tg_new
        feature_sup[count_new,:,:,:] = feature_new
        count_new += 1
        
    feat_sem = torch.cat((feat_sem, feat_sem_sup), 0)
    temp_sem = torch.cat((temp_sem, temp_sem_sup), 0)

    out_exc = net.classify(feature_exc)
    _, pred = torch.max(out_exc, 1)    
    out_sup = net.classify2(feature_sup)
    _, pred_sup = torch.max(out_sup, 1)
    
    target = target.cuda(async=True)
    target_exc = target_exc.cuda(async=True)
    target_sup = target_sup.cuda(async=True)
    
    loss_class = loss_class_func(out_exc, target_exc, out_sup=out_sup, target_sup=target_sup)
    loss_match = loss_match_func(feat_sem, temp_sem)
    loss_recon = loss_recon_func(recon_feat_sem, recon_temp_sem, template, recon_temp_sup=recon_temp_sup, template_sup=template_sup)
    loss_illu = loss_illu_func(feat_illu, target)
    loss = loss_class + beta*loss_match + gamma*loss_recon + loss_illu

    if i % 50 == 0:
      print('Epoch:%d  Batch:%d/%d  loss:%04f'%(e, i, batch_iter, loss/pred.numel()))
    
    f_trloss = open(os.path.join(result_path, "log_trloss.txt"),'a')
    f_trloss.write('Epoch:%d  Batch:%d/%d  loss:%04f\n'%(e, i, batch_iter, loss/pred.numel()))
    f_trloss.close()
    
    loss.backward()
    optimizer.step()
    
    if (e%save_epoch == 0 or e < 5):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)   

      if not out_root.is_dir():
        os.makedirs(out_folder, exist_ok=True)
      
      if i == show_batch or (e % 10 == 0 and i % 20 == 0):

        torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
        torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2) 
        torchvision.utils.save_image(recon_feat_sem, '{}/batch_{}_recon_feature.jpg'.format(out_folder,i), nrow=8, padding=2)
        torchvision.utils.save_image(recon_temp_sem, '{}/batch_{}_recon_template.jpg'.format(out_folder,i), nrow=8, padding=2)       
        
    pred_sup = pred_sup.cpu()
    target_sup = target_sup.cpu()
    right_idx = pred_sup == target_sup
    corr += (right_idx).sum().numpy()
    all += len(pred_sup)
    
  print('========epoch(%d)========='%e)
  print('Train accuracy')
  print('acc:%02f'%(corr/all))
  print('----------------------------------------------------------------')
  print('                                                                ')
  print('----------------------------------------------------------------')


def test(e, best_acc):
  net.eval()
  
  print('start test epoch: %d'%e)
  
  corr = 0
  all = 0
  feat_corr = 0
  feat_all = 0
  for i, (input, target, template) in enumerate(testloader):
    target = torch.squeeze(target)
    input, template = input.cuda(async=True), template.cuda(async=True)
    with torch.no_grad():
      feat_sem, feat_illu, feat_nowarp = net.extract(input, is_warping=True)
      temp_sem, _, _ = net.extract(template, is_warping=False)
      
      #decode
      recon_feat_sem = net.decode(feat_sem)
      recon_temp_sem = net.decode(temp_sem)
      
      #feature = mix_ratio * feat_sem + (1-mix_ratio) * feat_illu
      feature = torch.cat((feat_sem, feat_illu), 1)
      
      out = net.classify2(feature)
      _, pred = torch.max(out, 1)    
      target = target.cuda(async=True)
      
      loss_class = loss_class_func(out, target)
      loss_match = loss_match_func(feat_sem, temp_sem)
      loss_recon = loss_recon_func(recon_feat_sem, recon_temp_sem, template)
      loss_illu = loss_illu_func(feat_illu, target)
      loss = loss_class + beta*loss_match + gamma*loss_recon + loss_illu
      
      f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
      f_loss.write('Epoch:%d  Batch:%d/%d  loss:%04f\n'%(e, i, batch_iter, loss/pred.numel()))
      f_loss.close()
    
    if i % 5 == 0:
      print('Epoch:%d  Batch:%d/%d  loss:%04f'%(e, i, batch_iter_test, loss/pred.numel()))

    if (e%save_epoch == 0 or e < 5):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.makedirs(out_folder, exist_ok=True)
      
      if i == show_batch or (e % 10 == 0 and i % 20 == 0):  
        torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
        torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)     
        torchvision.utils.save_image(recon_feat_sem, '{}/batch_{}_recon_feature.jpg'.format(out_folder,i), nrow=8, padding=2)
        torchvision.utils.save_image(recon_temp_sem, '{}/batch_{}_recon_template.jpg'.format(out_folder,i), nrow=8, padding=2)
    
    # pred  
    pred = pred.cpu()
    target = target.cpu()
    right_idx = pred == target
    corr += (right_idx).sum().numpy()
    all += len(pred)

    if (e%save_epoch == 0 or e < 5):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.makedirs(out_folder, exist_ok=True)
      
      wi = ~right_idx  
      if i == show_batch:
        print('----------------------------------------------------------------')
        print(target[wi])
        print(pred[wi])
        
        
      if i%50 == 0:    
          f_miscls = open(os.path.join(result_path, "log_miscls.txt"),'a')
          f_miscls.write('Epoch:%d  Batch:%d/%d  targ:%s\n'%(e, i, batch_iter_test, str(target[wi])))
          f_miscls.write('Epoch:%d  Batch:%d/%d  pred:%s\n'%(e, i, batch_iter_test, str(pred[wi])))
          f_miscls.write('\n')
          f_miscls.close()
           
  print('========epoch(%d)========='%e)
  print('Test accuracy (beta=%s)'%(beta))
  print('acc:%02f'%(corr/all))
  print('                                                                ')
  print('================================================================')
  print('                                                                ')
  
  f_acc = open(os.path.join(result_path, "log_acc.txt"),'a')
  f_acc.write('Epoch:%d  acc:%02f\n'%(e, corr/all))
  f_acc.close()
  
  # Save models
  if best_acc < corr/all:
    best_acc = corr/all
    #save_time = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))
    torch.save(net, os.path.join(root_results, 'saved_models', (args.dataset + '_' + now + '.pth')))
      
  return best_acc


if __name__ == "__main__":
  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.makedirs(out_root, exist_ok=True)

  best_acc = 0

  for e in range(1, args.epochs + 1):
    train(e)
    best_acc = test(e,best_acc)
    
    print('========epoch(%d)========='%e)
    print('best_acc:%02f'%(best_acc))
