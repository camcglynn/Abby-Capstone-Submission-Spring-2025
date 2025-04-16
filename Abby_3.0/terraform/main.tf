provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "Abby"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

module "vpc" {
  source = "./modules/vpc"

  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
}

module "route53" {
  source = "./modules/route53"

  domain_name    = var.domain_name
  environment    = var.environment
  alb_dns_name   = module.alb.alb_dns_name
  alb_zone_id    = module.alb.alb_zone_id
  create_acm_certificate = true
}

module "ecr" {
  source = "./modules/ecr"

  environment = var.environment
  repository_names = [
    "abby-frontend",
    "abby-backend"
  ]
}

module "eks" {
  source = "./modules/eks"

  environment              = var.environment
  cluster_name             = var.cluster_name
  kubernetes_version       = var.kubernetes_version
  vpc_id                   = module.vpc.vpc_id
  private_subnet_ids       = module.vpc.private_subnet_ids
  public_subnet_ids        = module.vpc.public_subnet_ids
  node_group_instance_type = var.node_group_instance_type
  node_group_desired_size  = var.node_group_desired_size
  node_group_min_size      = var.node_group_min_size
  node_group_max_size      = var.node_group_max_size
}

module "alb" {
  source = "./modules/alb"

  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  certificate_arn    = module.route53.certificate_arn
  eks_cluster_name   = module.eks.cluster_name
}

module "redis" {
  source = "./modules/redis"

  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  redis_node_type    = var.redis_node_type
  security_group_ids = [module.eks.cluster_security_group_id]
}

module "rds" {
  source = "./modules/rds"

  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  db_instance_class   = var.db_instance_class
  security_group_ids  = [module.eks.cluster_security_group_id]
}

# Create S3 bucket for file uploads
resource "aws_s3_bucket" "uploads" {
  bucket = "${var.environment}-abby-uploads"

  tags = {
    Name = "${var.environment}-abby-uploads"
  }
}

# Block public access to the bucket
resource "aws_s3_bucket_public_access_block" "uploads_public_access_block" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning for the bucket
resource "aws_s3_bucket_versioning" "uploads_versioning" {
  bucket = aws_s3_bucket.uploads.id
  versioning_configuration {
    status = "Enabled"
  }
} 