import torch
import torch.nn.functional as F

def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    # ratio: [B,L]
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(
        teacher_probs * teacher_logprobs, inf_mask, 0
    )
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)  # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x - x) * mask.view(-1), dim=0) / torch.sum(
            mask.view(-1), dim=0
        )
    else:
        distil_loss = torch.sum(
            (teacher_x - x) * ratio.view(-1) * mask.view(-1), dim=0
        ) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


# For AKL
def get_akl_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(
        teacher_logits, torch.isinf(teacher_logits), 0
    ).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)  # B,L,V
    mask = cum_sum > mu
    mask[:, :, 0] = False  # 第一个概率一定要置False，对应第一个概率>0.5时mask全True

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1 / (s1 + s2), s2 / (s1 + s2)

# KL on top-k logits
def top_k_kl(teacher_logits, logits, mask, k=100):
    re_teacher_logits, idx = teacher_logits.topk(dim=-1, k=k)
    re_student_logits = logits.gather(dim=-1, index=idx)
    
    inf_mask = re_teacher_logits.isinf() | re_student_logits.isinf()
    distil_loss = get_jsd(re_teacher_logits, re_student_logits, inf_mask, mask)
    return distil_loss

def get_akl_loss(teacher_logits, logits, inf_mask, mask):
    h_ratio, t_ratio = get_akl_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(
        logits, teacher_logits, inf_mask, mask, t_ratio
    )
    return distil_loss


# Jensen-Shannon Divergence
def get_jsd(teacher_logits, logits, inf_mask, mask, beta=0.5):
    mixed_probs = beta * F.softmax(teacher_logits, dim=-1, dtype=torch.float32) + (
        1 - beta
    ) * F.softmax(logits, dim=-1, dtype=torch.float32)

    mixed_logprobs = torch.log(mixed_probs + 1e-8)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

    A = torch.masked_fill(
        teacher_probs * (teacher_logprobs - mixed_logprobs), inf_mask, 0
    )
    A = torch.sum(A, dim=-1).view(-1)
    A = torch.sum(A * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    B = torch.masked_fill(probs * (logprobs - mixed_logprobs), inf_mask, 0)
    B = torch.sum(B, dim=-1).view(-1)
    B = torch.sum(B * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    distil_loss = beta * A + (1 - beta) * B
    return distil_loss