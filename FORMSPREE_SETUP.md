# 📧 Formspree Setup Instructions

Your portfolio is now configured to use Formspree API for contact form submissions!

## 🚀 Quick Setup (2 minutes)

### Step 1: Create Formspree Account
1. Go to [https://formspree.io](https://formspree.io)
2. Sign up for a **FREE account** (no credit card required)
3. Verify your email address

### Step 2: Create New Form
1. Click **"+ New Form"** button
2. Name it: `Portfolio Contact Form`
3. Click **"Create Form"**

### Step 3: Get Your Form ID
After creating the form, you'll see a URL like:
```
https://formspree.io/f/xyzabc123
```

The part after `/f/` is your **Form ID** (e.g., `xyzabc123`)

### Step 4: Update Your Code
1. Open `index.html`
2. Find line **1242** (the form tag)
3. Replace `YOUR_FORM_ID` with your actual Form ID:

**BEFORE:**
```html
<form action="https://formspree.io/f/YOUR_FORM_ID" method="POST" ...>
```

**AFTER:**
```html
<form action="https://formspree.io/f/xyzabc123" method="POST" ...>
```

### Step 5: Test It!
1. Open your portfolio in a browser
2. Fill out the contact form
3. Submit it
4. Check your email - you'll receive the message!

## ⚙️ Optional Settings

In Formspree dashboard, you can customize:

- **Email Address**: Where messages are sent (default: your signup email)
- **Thank You Page**: Redirect users after submission
- **Notification Settings**: Email format, subject line
- **Spam Protection**: Already enabled by default! 🛡️
- **Submissions Archive**: View all messages in dashboard

## 📊 Free Plan Includes:
- ✅ **50 submissions/month**
- ✅ Unlimited forms
- ✅ Email notifications
- ✅ Spam filtering
- ✅ File uploads
- ✅ Submission archive

Perfect for a portfolio website! 🎉

## 🆘 Need Help?

If you get stuck:
1. Check Formspree docs: https://help.formspree.io
2. Verify your Form ID is correct
3. Make sure you're using `method="POST"`
4. Test with a simple submission first

---

**Note:** After first submission, Formspree will send a confirmation email. Click the link to activate the form permanently!
